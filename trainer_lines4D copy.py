import os
import os.path as osp
import logging
import torch
import torch.optim as optim
import numpy as np
import json
from tensorboardX import SummaryWriter
import gc

from lib.utils import load_model
from lib.file import ensure_dir
from lib.timer import *
from lib.loss import TotalLoss, MotorLoss
from lib.transformations import quaternion_from_matrix, project_G4_to_G3
from lib.ransac_l2l import run_ransac


class G4LinesTrainer:

  def __init__(self, config, data_loader, val_data_loader=None):

    # Model initialization
    Model = load_model("G4LinesRegression")
    #Model = load_model("PluckerNetRegression")
    self.model = Model(config)

    if config.weights:
      checkpoint = torch.load(config.weights)
      self.model.load_state_dict(checkpoint['state_dict'])

    logging.info(self.model)
    #
    self.config = config
    self.max_epoch = config.train_epoches
    self.save_freq = config.train_save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = np.inf

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    # limited GPU
    if config.gpu_inds > -1:
      torch.cuda.set_device(config.gpu_inds)
      self.device = torch.device('cuda', config.gpu_inds)
    else:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.optimizer = getattr(optim, config.optimizer)(self.model.parameters(), lr=config.train_lr,
        betas=(0.9, 0.999))

    #
    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)
    #
    self.start_epoch = config.train_start_epoch
    # concat dataset name
    self.checkpoint_dir = os.path.join(config.out_dir, config.dataset, config.model_nb)
    #
    ensure_dir(self.checkpoint_dir)
    json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'), indent=4, sort_keys=False)
    #
    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    self.test_valid = True if self.val_data_loader is not None else False
    self.model = self.model.to(self.device)
    self.writer = SummaryWriter(logdir=self.checkpoint_dir)
    #
    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)
        self.start_epoch = state['epoch']
        self.model.load_state_dict(state['state_dict'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])

        if 'best_val' in state.keys():
          self.best_val = state['best_val']
          self.best_val_epoch = state['best_val_epoch']
          self.best_val_metric = state['best_val_metric']
      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")

  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance


    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch()
      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self._save_checkpoint(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if self.best_val > val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, 'best_val_checkpoint')
        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )

  def _save_checkpoint(self, epoch, filename='checkpoint'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)


  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()


    # Main training
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_total_loss, batch_prob_matrix_loss = 0, 0
      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()

        #matches, line_bundle1, line_bundle2, R_gt, t_gt = data_loader_iter.next()
        matches, line_bundle1, line_bundle2, M_gt, R_gt, t_gt, _ , _ = next(data_loader_iter)

        data_time += data_timer.toc(average=False)


        # transfer all data to GPU
        matches, line_bundle1, line_bundle2, M_gt  = matches.to(self.device), line_bundle1.to(self.device), line_bundle2.to(self.device), M_gt.to(self.device)
        R_gt, t_gt = R_gt.to(self.device), t_gt.to(self.device)


        # Compute output
        #print(line_bundle1.shape, flush = True)

        M = self.model(line_bundle1, line_bundle2)

        M_gt = M_gt.view([-1, 8])
        M = M.view([-1, 8])

        #print(M_gt.shape, M.shape, flush = True)

        l1_lambda = 0.0001  # Regularization strength (tune this)
        l1_norm = 0.0

        # Sum the absolute values of all model parameters
        #for param in self.model.parameters():
        #    l1_norm += torch.sum(torch.abs(param))

        # compute the loss
        #MatchLoss = TotalLoss().to(self.device)
        Loss = MotorLoss().to(self.device)
        loss = Loss(M_gt, M) #+ l1_lambda * l1_norm

        

        #self.optimizer.zero_grad()
        loss.backward() # To accumulate gradient, zero gradients only at the begining of iter_size
        #self.optimizer.step()

        batch_total_loss += loss.item()

        # only used to monitor the training process
        #batch_prob_matrix_loss += ((1.0 - 2.0 * matches) * prob_matrix).sum(dim=(-2, -1)).mean()
        batch_prob_matrix_loss = 0

      self.optimizer.step()
      torch.cuda.empty_cache()

      total_loss += batch_total_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      # Print logs
      if curr_iter % self.config.print_freq == 0:
        self.writer.add_scalar('train/total_loss', batch_total_loss, start_iter + curr_iter)
        #self.writer.add_scalar('train/prob_loss', batch_prob_matrix_loss, start_iter + curr_iter)
        logging.info(
          "Train Epoch: {} [{}/{}], Current Training Loss: {:.3e}"
          .format(epoch, curr_iter,
                  len(self.data_loader) //
                  iter_size, batch_total_loss))
        data_meter.reset()
        total_timer.reset()

  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    num_data = 0
    data_timer, matching_timer = Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)

    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)

    data_loader_iter = self.val_data_loader.__iter__()

    # collecting the errors in rotation, errors in tranlsation, num of inliers, inlier ratios
    measure_list = ["err_q", "err_t", "inlier_ratio", "val_loss"]
    eval_res = {}
    for measure in measure_list:
        eval_res[measure] = np.zeros(tot_num_data)

    cnt = 0

    for batch_idx in range(tot_num_data):


      data_timer.tic()
      #matches, line_bundle1, line_bundle2, R_gt, t_gt = data_loader_iter.next()
      matches, line_bundle1, line_bundle2, M_gt, R_gt, t_gt, _, _ = next(data_loader_iter)

      data_timer.toc()

      nb_plucker = matches.size(1)

      # you can comment this line, as my GPU is short of memory
      #if nb_plucker > 3000 or nb_plucker < 2:
      #  continue

      matches, line_bundle1, line_bundle2 = matches.to(self.device), line_bundle1.to(self.device), line_bundle2.to(self.device)
      M = self.model(line_bundle1, line_bundle2)
      '''
      matching_timer.tic()
      #
      matching_timer.toc()

      # compute the topK correspondences
      k = min(100, round(line_bundle1.size(1) * line_bundle2.size(1)))  # Choose at most 100 points in the validation stage

      _, P_topk_i = torch.topk(prob_matrix.flatten(start_dim=-2), k=k, dim=-1, largest=True, sorted=True)

      line_bundle1_indices = P_topk_i / prob_matrix.size(-1)  # bxk (integer division)
      line_bundle2_indices = P_topk_i % prob_matrix.size(-1)  # bxk

      # in case cannot be estimated
      err_q = np.pi
      err_t = np.inf
      inlier_ratio = 0
      nb_inliers_gt = np.where(matches[0,:].cpu().numpy() > 0)[0].shape[0]
      # more than 3 3D-3D matches
      if k > 3:
        # let's check the inliner ratios within the topK matches
        # retrieve the inlier/outlier 1/0 logit

        #print(line_bundle1_indices, line_bundle2_indices, flush = True)
        inlier_inds = matches[:, line_bundle1_indices, line_bundle2_indices].cpu().numpy()
        inlier_ratio = np.sum(inlier_inds) / k * 100.0

        # compute the rotation and translation error

        line_bundle1_topK = line_bundle1[0, line_bundle1_indices[0, :k], :].cpu().numpy()
        line_bundle2_topK = line_bundle2[0, line_bundle2_indices[0, :k], :].cpu().numpy()

        if self.config.dataset == "structured3D" or self.config.dataset == "semantic3D":
          dis_threshold = 0.5
        else:
          dis_threshold = 1e-1

        best_rot, best_trans, best_ic, best_ic_mask = run_ransac(line_bundle1_topK.T, line_bundle2_topK.T, inlier_threshold = dis_threshold)

        if best_rot is None or best_trans is None:
          err_q, err_t = np.pi, np.inf
        else:
          err_q, err_t = self.evaluate_R_t(best_rot, best_trans, R_gt[0,:,:].numpy(), t_gt.numpy())
      '''   
      

      M_gt = M_gt.view([-1, 8])
      M = M.view([-1, 8])


      Loss = MotorLoss().to(self.device)
      val_loss = Loss(M_gt.to(self.device), M.to(self.device))
      
     
      if cnt < 6:
        print(M_gt, flush = True)
        print(M, flush = True)
        print("----")
        cnt += 1
      #print("*****")

      err_q, err_t = self.evaluate_R_t(R_gt[0,:,:].numpy(), t_gt.numpy(), M.detach().cpu().numpy(), M_gt.detach().cpu().numpy())

      num_data += 1
      torch.cuda.empty_cache()

      eval_res["err_q"][batch_idx] = err_q
      eval_res["err_t"][batch_idx] = err_t
      eval_res["val_loss"][batch_idx] = val_loss
      #eval_res["inlier_ratio"][batch_idx] = inlier_ratio

      logging.info(' '.join([
          f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
          f"err_rot: {err_q:.3f}, err_t: {err_t:.3f}, val_loss: {val_loss:.5f}",
      ]))
      data_timer.reset()

    # after checking all samples, let's calculate statistics
    recall = self.recalls(eval_res)

    logging.info(' '.join([
        f"recall_rot: {recall[0]:.3f}, mean rot. : {recall[1]:.3f}, mean trans. : {recall[2]:.3f}, avg. inlier ratio: {recall[3]:.3f},",
    ]))
    logging.info(' '.join([
        f"med rot. : {recall[4]:.3f}, med trans. : {recall[5]:.3f}, mean_val_loss: {recall[6]:.8f},",
    ]))
  
    return {
        "recall_rot": recall[0],
        "mean_rot": recall[1],
        "mean_trans": recall[2],
        "avg_inlier_ratio": recall[3],
        "med_rot": recall[4],
        "med_trans": recall[5],
        "mean_val_loss": recall[6],
    }



  def evaluate_R_t(self, R_gt, t_gt, M, M_gt, q_gt=None):
    
    #print(M, flush = True)
    R_est, t_est, _ , _ = project_G4_to_G3(M, t_gt)
    R_gt, t_gt0,  _ , _ = project_G4_to_G3(M_gt, t_gt)


    '''
    print(t_gt, flush = True)
    print(t_gt0, flush = True)
    print("****")
    
    print(R_gt, flush = True)
    print(R_gt0, flush = True)
    print("****")
    print(M, flush = True)
    print(M_gt, flush = True)
    '''


    #t = t_est.flatten()
    t_gt = t_gt0.flatten()
    t = t_est.flatten()

    eps = 1e-15

    R_est = np.reshape(R_est, (3, 3))
    R_gt = np.reshape(R_gt, (3, 3))

    if q_gt is None:
      q_gt = quaternion_from_matrix(R_gt)

    q = quaternion_from_matrix(R_est)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    #print("---")
    #print(q)
    #print(q_gt)
    #print("---")
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)
    # absolute distance error on t
    #err_t = np.linalg.norm(t_gt - t)
    err_t = np.sqrt(np.sum((t_gt - t) ** 2))
    '''
    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
      # This should never happen! Debug here
      err_q = np.pi
      err_t = np.inf
    '''
    return err_q, err_t

  def recalls(self, eval_res):

    ret_val = []
    ths = np.arange(7) * 5
    cur_err_q = np.array(eval_res["err_q"]) * 180.0 / np.pi
    # Get histogram
    q_acc_hist, _ = np.histogram(cur_err_q, ths)
    num_pair = float(len(cur_err_q))
    q_acc_hist = q_acc_hist.astype(float) / num_pair
    q_acc = np.cumsum(q_acc_hist)
    # Store return val
    ret_val += [np.mean(q_acc[:4])]
    ret_val += [np.mean(cur_err_q)]
    ret_val += [np.mean(eval_res["err_t"])]
    ret_val += [np.mean(eval_res["inlier_ratio"])]
    ret_val += [np.median(cur_err_q)]
    ret_val += [np.median(eval_res["err_t"])]
    ret_val += [np.mean(eval_res["val_loss"])]

    return ret_val
