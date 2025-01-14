import os
import sys
import numpy as np
from torch.utils.data import Dataset
import pickle
from clifford.g3c import *
from math import sqrt
import numpy as np

def load_data_plucker_pairs(config, dataset_split):
    """Main data loading routine"""
    print("loading the dataset {} ....\n".format(config.dataset))

    var_name_list = ["matches", "plucker1", "plucker2", "R_gt", "t_gt"]

    # check system python version
    if sys.version_info[0] == 3:
        print("You are using python 3.")

    encoding = "latin1"
    # Let's unpickle and save data
    data = {}
    # load the data

    cur_folder = "/".join([config.data_dir, config.dataset + "_" + dataset_split])
    for var_name in var_name_list:

        if config.dataset == "scenecity3D" and dataset_split == "train":
            # this large dataset has two partitions
            in_file_names = [os.path.join(cur_folder, var_name) + "_part1.pkl", os.path.join(cur_folder, var_name) + "_part2.pkl"]

            for in_file_name in in_file_names:
                with open(in_file_name, "rb") as ifp:
                    if var_name in data:
                        if sys.version_info[0] == 3:
                            data[var_name] += pickle.load(ifp, encoding=encoding)
                        else:
                            data[var_name] += pickle.load(ifp)
                    else:
                        if sys.version_info[0] == 3:
                            data[var_name] = pickle.load(ifp, encoding=encoding)
                        else:
                            data[var_name] = pickle.load(ifp)
        else:
            in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    if sys.version_info[0] == 3:
                        data[var_name] += pickle.load(ifp, encoding=encoding)
                    else:
                        data[var_name] += pickle.load(ifp)
                else:
                    if sys.version_info[0] == 3:
                        data[var_name] = pickle.load(ifp, encoding=encoding)
                    else:
                        data[var_name] = pickle.load(ifp)
    print("[Done] loading the {} dataset of  {} ....\n".format(dataset_split, config.dataset))

    return data


def plucker_to_points(plucker1):
    # Split the Plücker coordinates into the direction vector (L) and moment vector (M)
    L1 = np.array(plucker1[:,:3])  # (L_x, L_y, L_z)
    M1 = np.array(plucker1[:,3:])  # (M_x, M_y, M_z)

    # Check if the direction vector L is valid (should not be zero)
    if np.allclose(L1, 0):
        raise ValueError("Invalid Plücker coordinates: direction vector L is zero.")
    
    # Calculate |L|^2

    #print("SHAPES:", plucker1.shape, L1.shape, flush=True)

    #print(L1.shape, flush = True)

    L_norm_sq = (np.sqrt(np.sum(L1**2, axis=1))).reshape(-1, 1)  # shape (A, 1) for broadcasting
    
    # Compute a point P1 on the line
    P1 = np.cross(M1, L1) / L_norm_sq
    
    # Choose a second point P2 on the line by adding the direction vector L to P1
    P2 = P1 + L1  # or P2 = P1 + t*L for any t (here, t = 1)

    return P1, P2




#From Euclidean to 1D Up CGA. function implementing the Eq. 10 (X = f(x))
def up1D(point, lambd = 200):
    x = point[:,0]*e1 + point[:,1]*e2 + point[:,2]*e3
    Y = (2*lambd / (lambd**2 + x**2))*x + ((lambd**2-x**2)/(lambd**2 + x**2))*e4
    return Y

def form_4Dlines(P, Q):

    L =  P ^ Q
    #L = L / (L)

    #print(L.shape, flush = True)

    p = []
    q = []
    r = []
    s = []
    t = []
    u = []

    for l in L:
        #normalize the line
        l = l / sqrt((l * ~l)[0])

        #extract bivector coefficients
        p.append(l[6])
        q.append(l[7])
        r.append(l[8])
        s.append(l[10])
        t.append(l[11])
        u.append(l[13])


    p = np.asarray(p).reshape(-1, 1)
    q = np.asarray(q).reshape(-1, 1)
    r = np.asarray(r).reshape(-1, 1)
    s = np.asarray(s).reshape(-1, 1)
    t = np.asarray(t).reshape(-1, 1)
    u = np.asarray(u).reshape(-1, 1)

    #print((L[:][13]).shape, flush = True)
    #print(u.shape, flush = True)
    #print(u, flush = True)
    #print("***", flush = True)
    #print(L, flush = True)

    L = np.concatenate([p, q, r, s, t, u], axis = 1)

    return L

def form_motors(R, t, lambd = 200):

    #assign translation vector components to a basis
    x = t[0]*e1 + t[1]*e2 + t[2]*e3

    #translation rotor in G4
    T = (lambd + x*e4)/(sqrt(lambd**2 + x**2))

    #assign rotation matrix components to a basis
    B = [R[0,0]*e1 + R[1,0]*e2 + R[2,0]*e3,
     R[0,1]*e1 + R[1,1]*e2 + R[2,1]*e3,
     R[0,2]*e1 + R[1,2]*e2 + R[2,2]*e3]

    #print(R.shape)
    #print(t.shape)
    
    A = [e1,e2,e3]
    R = 1+sum([A[k]*B[k] for k in range(3)])
    R = R/abs(R)

    M = R*T

    M = M / sqrt((M * ~M)[0])

    #print(M, flush = True)
    #print(M[0], flush = True)

    m = []

    m.append(M[0][0])
    m.append(M[0][6])
    m.append(M[0][7])
    m.append(M[0][8])
    m.append(M[0][10])
    m.append(M[0][11])
    m.append(M[0][13])
    m.append(M[0][26])

    #print(m)



    return np.asarray(m).reshape(-1, 1)


# This is loading the pre_dumped dataset
class Lines4D_precompute(Dataset):
    def __init__(self, phase, config):
        super(Lines4D_precompute, self).__init__()
        self.phase = phase
        self.config = config
        self.data = load_data_plucker_pairs(config, phase)
        self.len = len(self.data["t_gt"])

    def __getitem__(self, index):
        matches_ind = self.data["matches"][index]
        plucker1 = self.data["plucker1"][index]
        plucker2 = self.data["plucker2"][index]
        R_gt = self.data["R_gt"][index]
        t_gt = self.data["t_gt"][index]

        nb_lines1 = plucker1.shape[0]
        nb_lines2 = plucker2.shape[0]

        #print("SHAPES:", plucker1.shape, plucker2.shape, R_gt.shape, t_gt.shape, flush=True)

        p1, q1 = plucker_to_points(plucker1)
        p2, q2 = plucker_to_points(plucker2)

        #print(p1.shape, q1.shape, flush = True)

        P1 = up1D(p1)
        Q1 = up1D(q1)

        #print(P1.shape, Q1.shape,  flush = True)

        P2 = up1D(p2)
        Q2 = up1D(q2)

        L1 = form_4Dlines(P1, Q1)
        L2 = form_4Dlines(P2, Q2)

        #print(L1.shape, L2.shape, flush = True)


        M_gt = form_motors(R_gt, t_gt)

        #print(M_gt.shape)

        #print(L1)
        #print(plucker1)


        matches = np.zeros([nb_lines1, nb_lines2], dtype=np.float32)
        matches[matches_ind[0,:], matches_ind[1,:]] = 1.0


        return matches.astype('float32'), L1.astype('float32'), L2.astype('float32'), M_gt.astype('float32'), R_gt.astype('float32'), t_gt.astype('float32')

    def __len__(self):
        return len(self.data["t_gt"])












