import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math
from torchga.torchga import GeometricAlgebra
from torchga.layers import TensorToGeometric, GeometricToTensor, GeometricProductDense, GeometricSandwichProductDense, GeometricProductConv1D
from cgenn.algebra.cliffordalgebra import CliffordAlgebra
from cgenn.models.modules.gp import SteerableGeometricProductLayer
from cgenn.models.modules.linear import MVLinear
from cgenn.models.modules.mvlayernorm import MVLayerNorm
from cgenn.models.modules.mvsilu import MVSiLU

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=10, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=min(k,num_points))  # (batch_size, num_points, k)
    device = torch.device('cuda')

    nb_knns = idx.size(-1)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, nb_knns, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, nb_knns, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous() # (batch_size, num_dims*2, num_points, k)
    return feature


class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def MLP(channels: list, do_gn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_gn:
                layers.append(nn.GroupNorm(4, channels[i]))
            layers.append(GELU_())
    return nn.Sequential(*layers)


# calculate the pairwise distance for plucker features
def pairwiseL2Dist(x1, x2):
    """ Computes the pairwise L2 distance between batches of feature vector sets
    res[..., i, j] = ||x1[..., i, :] - x2[..., j, :]||
    since
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

    Adapted to batch case from:
        jacobrgardner
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm2 = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm2.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm2).clamp_min_(1e-30).sqrt_()
    return res


# Sinkhorn to estimate the joint probability matrix P
class prob_mat_sinkhorn(torch.nn.Module):
    def __init__(self, config, mu=0.1, tolerance=1e-9, iterations=30):
        super(prob_mat_sinkhorn, self).__init__()
        self.config = config
        self.mu = mu  # the smooth term
        self.tolerance = tolerance  # don't change
        self.iterations = iterations  # max 30 is set, enough for a typical sized mat (e.g., 1000x1000)
        self.eps = 1e-12

    def forward(self, M, r=None, c=None):
        # r, c are the prior 1D prob distribution of pluecker lines
        # M is feature distance between source and target lines
        K = (-M / self.mu).exp()
        # 1. normalize the matrix K
        K = K / K.sum(dim=(-2, -1), keepdim=True).clamp_min_(self.eps)

        # 2. construct the unary prior

        r = r.unsqueeze(-1)
        u = r.clone()
        c = c.unsqueeze(-1)

        i = 0
        u_prev = torch.ones_like(u)
        while (u - u_prev).norm(dim=-1).max() > self.tolerance:
            if i > self.iterations:
                break
            i += 1
            u_prev = u
            # update the prob vector u, v iteratively
            v = c / K.transpose(-2, -1).matmul(u).clamp_min_(self.eps)
            u = r / K.matmul(v).clamp_min_(self.eps)

        # assemble
        # P = torch.diag_embed(u[:,:,0]).matmul(K).matmul(torch.diag_embed(v[:,:,0]))
        P = (u * K) * v.transpose(-2, -1)
        return P


class conv_in_seq_direction_moment_knn(nn.Module):
    def __init__(self, out_channel: int):
        super().__init__()
        self.in_channel = 6
        self.seq_out_channel = out_channel//2

        self.conv_direction = torch.nn.Conv2d(self.in_channel, self.seq_out_channel // 8, 1)
        self.conv_moment = torch.nn.Conv2d(self.in_channel, self.seq_out_channel // 8, 1)

        self.mlp_direction = MLP([ self.seq_out_channel // 8, self.seq_out_channel // 4,  self.seq_out_channel // 2, self.seq_out_channel])
        self.mlp_moment = MLP([ self.seq_out_channel // 8, self.seq_out_channel // 4,  self.seq_out_channel // 2, self.seq_out_channel])

        self.mlp_merged = MLP([out_channel, out_channel, out_channel])

    def forward(self, x):
        # for each direction, find it's knn feature
        x_knn_direction = self.conv_direction(get_graph_feature (x[:,:3,:])).mean(dim=-1, keepdim=False)
        x_knn_moment = self.conv_moment(get_graph_feature(x[:, 3:, :])).mean(dim=-1, keepdim=False)

        x_direction = self.mlp_direction(x_knn_direction)
        x_moment = self.mlp_moment(x_knn_moment)

        x_concat = torch.cat([x_direction, x_moment], dim=-2)
        x_concat = self.mlp_merged(x_concat)
        return x_concat

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]

        # original softmax attention
        x, prob = attention(query, key, value)

        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))



class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))



class SpatialAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names
        self.mlp = MLP([feature_dim * 3,  feature_dim * 2,  feature_dim * 2, feature_dim])

    def forward(self, desc0, desc1):

        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1

            delta0 = layer(desc0, src0)
            delta1 = layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

        # given the updated feature, let's add MLP to regress per-point prior matchability
        desc0_detached = desc0
        desc1_detached = desc1
        # desc0 will concat the global feature from desc1 and vice versa

        desc0_global = torch.cat((desc0_detached.mean(dim=-1, keepdim=True), desc0_detached.max(dim=-1, keepdim=True)[0]), dim=-2).repeat(1, 1, desc1_detached.size(-1))
        desc1_global = torch.cat((desc1_detached.mean(dim=-1, keepdim=True), desc1_detached.max(dim=-1, keepdim=True)[0]), dim=-2).repeat(1, 1, desc0_detached.size(-1))
        desc0_regress = torch.cat((desc0_detached, desc1_global), dim=-2)
        desc1_regress = torch.cat((desc1_detached, desc0_global), dim=-2)

        # project the feature_regress
        desc0_regress = self.mlp(desc0_regress)
        desc1_regress = self.mlp(desc1_regress)

        return desc0, desc1, desc0_regress, desc1_regress

# use graph net to extract features
class FeatureExtractorGraph(nn.Module):

  def __init__(self, config, in_channel):

    super(FeatureExtractorGraph, self).__init__()

    self.config = config

    self.regress = nn.Conv1d(self.config['net_nchannel'], 1, kernel_size=1, bias=True)
    self.gnn = SpatialAttentionalGNN(self.config['net_nchannel'], self.config['GNN_layers'])
    self.final_proj = nn.Conv1d(self.config['net_nchannel'], self.config['net_nchannel'], kernel_size=1, bias=True)
    self.conv_in = conv_in_seq_direction_moment_knn(self.config['net_nchannel'])

  def forward(self, x, y):
      # Multi-layer Transformer network.
      desc0, desc1, x_prob, y_prob = self.gnn(self.conv_in(x), self.conv_in(y))
      # Final MLP projection.
      mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
      # ---------------------------------------------------------------------------------
      # Final MLP projection.
      x_prob_logits = self.regress(x_prob)
      y_prob_logits = self.regress(y_prob)
      # ---------------------------------------------------------------------------------
      # perform softmax to obtain unary matching prior
      x_prob = x_prob_logits.softmax(dim=-1)
      y_prob = y_prob_logits.softmax(dim=-1)
      # ---------------------------------------------------------------------------------

      return mdesc0, mdesc1, x_prob, y_prob


# the main plucker net
class PluckerNetKnn(nn.Module):
    def __init__(self, config):
        super(PluckerNetKnn, self).__init__()
        self.config = config
        self.in_channel = 6  # the number of dimensions for plucker line
        # feature extractor
        self.FeatureExtractor = FeatureExtractorGraph(self.config, self.in_channel)
        # calculate the pairwise distance for plucker features
        self.pairwiseL2Dist = pairwiseL2Dist
        # configurations for the estimation of joint probability matrix
        self.sinkhorn_mu = config.net_lambda
        self.sinkhorn_tolerance = 1e-9
        self.iterations = config.net_maxiter
        self.sinkhorn = prob_mat_sinkhorn(self.config, self.sinkhorn_mu, self.sinkhorn_tolerance, self.iterations)

    def forward(self, plucker1, plucker2):
        # Extract features with line-wise probability
        plucker1_feats, plucker2_feats, plucker1_prob, plucker2_prob = self.FeatureExtractor(plucker1.transpose(-2, -1), plucker2.transpose(-2, -1))
        plucker1_feats = plucker1_feats.transpose(-2, -1)  # b x n x 128
        plucker2_feats = plucker2_feats.transpose(-2, -1)  # b x n x 128
        # L2 Normalise:
        plucker1_feats = torch.nn.functional.normalize(plucker1_feats, p=2, dim=-1)
        plucker2_feats = torch.nn.functional.normalize(plucker2_feats, p=2, dim=-1)
        # Compute pairwise L2 distance matrix:
        # row : plucker1 index; col : plucker2 index
        M = self.pairwiseL2Dist(plucker1_feats, plucker2_feats)
        # Sinkhorn:
        r = plucker1_prob.squeeze(1)
        c = plucker2_prob.squeeze(1)

        P = self.sinkhorn(M, r, c)

        return P, r, c


# --------------------------------------------------------------------------

# the direct regression network
# directly output a quat. and trans. WITHOUT estimating line-to-line matches

class Pooling(torch.nn.Module):
    def __init__(self, pool_type='max'):
            self.pool_type = pool_type
            super(Pooling, self).__init__()

    def forward(self, input):
        if self.pool_type == 'max':
            return torch.max(input, 2)[0].contiguous()
        elif self.pool_type == 'avg' or self.pool_type == 'average':
            return torch.mean(input, 2).contiguous()


class PluckerNetRegression(nn.Module):
    def __init__(self, config):
        super(PluckerNetRegression, self).__init__()
        self.config = config
        self.in_channel = 6  # the number of dimensions for plucker line
        self.FeatureExtractor = FeatureExtractorGraph(self.config, self.in_channel)
        self.linear = [nn.Linear(config.net_nchannel*2, config.net_nchannel*2), nn.ReLU(),
                       nn.Linear(config.net_nchannel*2, config.net_nchannel), nn.ReLU(),
                       nn.Linear(config.net_nchannel, config.net_nchannel), nn.ReLU(),
                       nn.Linear(config.net_nchannel, config.net_nchannel // 2), nn.ReLU(),
                       nn.Linear(config.net_nchannel // 2, config.net_nchannel // 2), nn.ReLU()]

        # quat and trans
        self.linear.append(nn.Linear(config.net_nchannel // 2, 8))
        self.linear = nn.Sequential(*self.linear)
        self.pooling = Pooling('max')

    def create_pose(self, M):
        # Normalize the motor

        ga = GeometricAlgebra([1, 1, 1, 1])
        columns_to_select = [0, 5, 6, 7, 8, 9, 10, 15]
        even_indices = torch.tensor(columns_to_select)
        

        M = ga.from_tensor(M, blade_indices=even_indices)
        Minv = ga.reversion(M)

        scalar = (ga.geom_prod(M, Minv))[:,0]

        M = M[:, columns_to_select]

        M = M / torch.sqrt(scalar.view(-1, 1) + 1e-8)

        return M.view([-1, 8])


    def forward(self, plucker1, plucker2, r = None, c = None):
        # Extract features

        #print(plucker1.shape, flush = True)
        plucker1_feats, plucker2_feats, _, _ = self.FeatureExtractor(plucker1.transpose(-2, -1), plucker2.transpose(-2, -1))
        plucker1_feats, plucker2_feats = self.pooling(plucker1_feats), self.pooling(plucker2_feats)
        plucker_feats_cat = torch.cat([plucker1_feats, plucker2_feats], dim=1)
        pose = self.linear(plucker_feats_cat)
        pose = self.create_pose(pose)
        #print(pose.shape, flush = True)
        return pose


class O4CGMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0, 1.0))

        # Second block (256 -> 128)
        self.linear2 = MVLinear(self.algebra, 32, 128, subspaces=False)
        self.block2 = nn.Sequential(
            MVSiLU(self.algebra, 128),
            SteerableGeometricProductLayer(self.algebra, 128),
            MVLayerNorm(self.algebra, 128),
        )

        # Third block (128 -> 40)
        self.linear3 = MVLinear(self.algebra, 128, 40, subspaces=False)
        self.block3 = nn.Sequential(
            MVSiLU(self.algebra, 40),
            SteerableGeometricProductLayer(self.algebra, 40),
            MVLayerNorm(self.algebra, 40),
        )


        self.linear4 = MVLinear(self.algebra, 40, 1, subspaces=False)
        self.block4 = nn.Sequential(
            MVSiLU(self.algebra, 1),
            SteerableGeometricProductLayer(self.algebra, 1),
            MVLayerNorm(self.algebra, 1),
        )

    def forward(self, x):
     
        x = self.linear2(x)
        x = self.block2(x)
        x = self.linear3(x)
        x = self.block3(x)
        x = self.linear4(x)
        x = self.block4(x)


        return x


    
class PoolingFinal(torch.nn.Module):
    def __init__(self, pool_type='max'):
            self.pool_type = pool_type
            super(PoolingFinal, self).__init__()

    def forward(self, input):
        if self.pool_type == 'max':
            return torch.max(input, 1)[0].contiguous()
        elif self.pool_type == 'avg' or self.pool_type == 'average':
            return torch.mean(input, 1).contiguous()
        

class G4LinesRegression(nn.Module):
    def __init__(self, config):
        super(G4LinesRegression, self).__init__()

        self.config = config
        self.in_channel = 6
    
        self.pooling = Pooling('max')
        self.pooling_final = PoolingFinal('avg')

        self.ga = GeometricAlgebra([1,1,1,1])
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0, 1.0))

        self.biv_indices = torch.tensor([5, 6, 7, 8, 9, 10])
        self.even_indices = torch.tensor([0, 5, 6, 7, 8, 9, 10, 15])
        self.all = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        
        self.tensor_to_geometric_lines = TensorToGeometric(self.ga, blade_indices=self.biv_indices)
        self.geometric_to_tensor_poses = GeometricToTensor(self.ga, blade_indices=self.even_indices)

        self.FeatureExtractor = FeatureExtractorGraph(self.config, self.in_channel)


        self.o4mlp = O4CGMLP()

        self.sp1 = GeometricSandwichProductDense(self.ga, 32, 128, 
                                                 activation = None,
                                                 blade_indices_kernel=self.even_indices, 
                                                 blade_indices_bias=self.even_indices)
        self.sp2 = GeometricSandwichProductDense(self.ga, 128, 40, 
                                                 activation = None,
                                                 blade_indices_kernel=self.even_indices, 
                                                 blade_indices_bias=self.even_indices)
    
        self.sp3 = GeometricSandwichProductDense(self.ga, 40, 1, 
                                                 activation = None,
                                                 blade_indices_kernel=self.even_indices, 
                                                 blade_indices_bias=self.even_indices)
        
    

        self.act = nn.Identity()
        self.act1 = MVSiLU(self.algebra, 128)
        self.act2 = MVSiLU(self.algebra, 40)
        self.act3 = MVSiLU(self.algebra, 32)



    def create_lines(self, l):
        # Normalize the motor

        ga = GeometricAlgebra([1, 1, 1, 1])
        columns_to_select = [5, 6, 7, 8, 9, 10]
        biv_indices = torch.tensor(columns_to_select)
        

        l = ga.from_tensor(l, blade_indices=biv_indices)
        linv = ga.reversion(l)

        scalar = (ga.geom_prod(l, linv))[:,:,0]

        l = l[:, :, columns_to_select]
        l = l / torch.sqrt(scalar.view(scalar.shape[0], scalar.shape[1], 1) + 1e-8)

        return l.view(scalar.shape[0], scalar.shape[1], 6)
    
    def create_pose(self, M):
        # Normalize the motor

        ga = GeometricAlgebra([1, 1, 1, 1])
        columns_to_select = [0, 5, 6, 7, 8, 9, 10, 15]
        even_indices = torch.tensor(columns_to_select)
        

        M = ga.from_tensor(M, blade_indices=even_indices)
        Minv = ga.reversion(M)

        scalar = (ga.geom_prod(M, Minv))[:,0]

        M = M[:, columns_to_select]

        M = M / torch.sqrt(scalar.view(-1, 1) + 1e-8)

        return M.view([-1, 8])


    
    def forward(self, lines1, lines2):
        # Extract features

        #reshape in B x N x c_in x 6


        l1_feats, l2_feats, _, _ = self.FeatureExtractor(lines1.transpose(-2, -1), lines2.transpose(-2, -1))
        l1_feats, l2_feats = self.pooling(l1_feats), self.pooling(l2_feats)
        #l_feats_cat = torch.cat([l1_feats, l2_feats], dim=1)
        #l_feats_cat = 0.5*(l1_feats + l2_feats)

        #l_feats_cat = l_feats_cat.unsqueeze(2)

        l1_feats= l1_feats.reshape((-1, 32, 6))
        l2_feats= l2_feats.reshape((-1, 32, 6))


        l1 = self.create_lines(l1_feats)
        l2 = self.create_lines(l2_feats)


        l1 = self.tensor_to_geometric_lines(l1)
        l2 = self.tensor_to_geometric_lines(l2)


        x1 = self.act(self.sp1(l1))
        x2 = self.act(self.sp2(x1)) 
        out_sp1 = self.act(self.sp3(x2))

        x3 = self.act(self.sp1(l2)) + x1
        x4 = self.act(self.sp2(x3)) + x2
        out_sp2 = self.act(self.sp3(x4))

        

        out_eq1 = self.o4mlp(l1)
        out_eq2 = self.o4mlp(l2)
          
        out = torch.cat([out_sp1, out_eq1, out_sp2, out_eq2], dim=1)
        out = self.pooling_final(out)

        pose = self.geometric_to_tensor_poses(out)
        #pose = pose.squeeze(dim = 1)
        pose = self.create_pose(pose)
       
        return pose






