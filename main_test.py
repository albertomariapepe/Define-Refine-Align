
import os
from easydict import EasyDict as edict
import logging
import sys
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader


from config import get_config
from lib.utils import load_model
from lib.timer import *
from lib.ransac_l2l import run_ransac
from lib.transformations import quaternion_from_matrix, project_G4_to_G3
from lib.dataloader import Lines4D_precompute

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

logging.basicConfig(level=logging.INFO, format="")


def evaluate_R_t(R_gt, t_gt, M, M_gt, q_gt=None):
    
    #print(M, flush = True)
    R_est, t_est, _ , _  = project_G4_to_G3(M, t_gt)
    R_gt, t_gt, _ , _  = project_G4_to_G3(M_gt, t_gt)

    t = t_est.flatten()
    t_gt = t_gt.flatten()
    eps = 1e-15

    if q_gt is None:
      q_gt = quaternion_from_matrix(R_gt)


    #print(R_gt)
    R_est = np.reshape(R_est, (3, 3))

    #print(R_est)
    #print("***")

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
    err_t = np.linalg.norm(t_gt - t)
    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
      # This should never happen! Debug here
      err_q = np.pi
      err_t = np.inf
    return err_q, err_t

def recalls(eval_res):

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
    ret_val += [np.quantile(cur_err_q, 0.25)]
    ret_val += [np.quantile(cur_err_q, 0.75)]
    ret_val += [np.quantile(eval_res["err_t"], 0.25)]
    ret_val += [np.quantile(eval_res["err_t"], 0.75)]
    #ret_val += [np.mean(eval_res["val_loss"])]

    return ret_val


# main function

def main(config):

    val_data_loader = DataLoader(Lines4D_precompute(phase='valid', config = configs), batch_size=1, shuffle=False, drop_last=False, num_workers=1)
    # no gradients
    with torch.no_grad():
        # Model initialization
        Model = load_model("G4LinesRegression")
        #Model = load_model("PluckerNetRegression")
        model = Model(config)

        # limited GPU
        if config.gpu_inds > -1:
            torch.cuda.set_device(config.gpu_inds)
            device = torch.device('cuda', config.gpu_inds)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)

        # load the weights
        print(config.weights, flush = True)
        if config.weights:
            print("Loading pretrained model...")
            checkpoint = torch.load(config.weights)
            model.load_state_dict(checkpoint['state_dict'])

        logging.info(model)


        # evaluation model
        model.eval()

        num_data = 0
        data_timer, matching_timer = Timer(), Timer()
        tot_num_data = len(val_data_loader.dataset)

        data_loader_iter = val_data_loader.__iter__()

        # collecting the errors in rotation, errors in tranlsation, num of inliers, inlier ratios
        measure_list = ["err_q", "err_t", "inlier_ratio"]
        eval_res = {}
        for measure in measure_list:
            eval_res[measure] = np.zeros(tot_num_data)


        counter = 0

        for batch_idx in range(tot_num_data):

            data_timer.tic()
            #matches, plucker1, plucker2, R_gt, t_gt = data_loader_iter.next()
            matches, line_bundle1, line_bundle2, M_gt, R_gt, t_gt, plucker1, plucker2 = next(data_loader_iter)

            matches, line_bundle1, line_bundle2, M_gt  = matches.to(device), line_bundle1.to(device), line_bundle2.to(device), M_gt.to(device)
            #R_gt, t_gt = R_gt.to(device), t_gt.to(device)

            M = model(line_bundle1, line_bundle2)

            M_gt = M_gt.view([-1, 8])
            M = M.view([-1, 8])

            data_timer.toc()
            '''
            nb_plucker = matches.size(1)

            # you can comment this line, as my GPU is short of memory
            if nb_plucker > 3000 or nb_plucker < 2:
                continue


            matches, plucker1, plucker2  = matches.to(device), plucker1.to(device), plucker2.to(device)

            # Compute output
            matching_timer.tic()
            prob_matrix, prior1, prior2 = model(plucker1, plucker2)
            matching_timer.toc()

            # compute the topK correspondences
            k = min(200, round(plucker1.size(1) * plucker2.size(1)))
            _, P_topk_i = torch.topk(prob_matrix.flatten(start_dim=-2), k=k, dim=-1, largest=True, sorted=True)

            plucker1_indices = P_topk_i / prob_matrix.size(-1)  # bxk (integer division)
            plucker2_indices = P_topk_i % prob_matrix.size(-1)  # bxk

            # in case cannot be estimated
            err_q = np.pi
            err_t = np.inf
            inlier_ratio = 0
            nb_inliers_gt = np.where(matches[0, :].cpu().numpy() > 0)[0].shape[0]
            # more than 3 3D-3D matches
            if k > 3:
                # let's check the inliner ratios within the topK matches
                # retrieve the inlier/outlier 1/0 logit
                inlier_inds = matches[:, plucker1_indices, plucker2_indices].cpu().numpy()
                inlier_ratio = np.sum(inlier_inds) / k * 100.0

                # compute the rotation and translation error
                plucker1_topK = plucker1[0, plucker1_indices[0, :k], :].cpu().numpy()
                plucker2_topK = plucker2[0, plucker2_indices[0, :k], :].cpu().numpy()

                if config.dataset == "structured3D" or config.dataset == "semantic3D":
                    dis_threshold = 0.5
                else:
                    dis_threshold = 1e-1
                best_rot, best_trans, best_ic, best_ic_mask = run_ransac(plucker1_topK.T, plucker2_topK.T, inlier_threshold=dis_threshold)

                if best_rot is None or best_trans is None:
                    err_q, err_t = np.pi, np.inf
                else:
                    err_q, err_t = evaluate_R_t(best_rot, best_trans, R_gt[0, :, :].numpy(), t_gt.numpy())
            '''
            
            if counter < 10:
                path = os.path.join(configs.out_dir, configs.dataset, configs.model_nb) + "/motors"
                np.save(path + "/line_bundle_1_" + str(counter) + ".npy", plucker1)
                np.save(path + "/line_bundle_2_" + str(counter) + ".npy", plucker2)
                np.save(path + "/motor_GT_" + str(counter) + ".npy", M_gt.detach().cpu().numpy())
                np.save(path + "/motor_pred_" + str(counter) + ".npy", M.detach().cpu().numpy())
                np.save(path + "/t_GT_" + str(counter) + ".npy", t_gt)

                counter += 1



            err_q, err_t = evaluate_R_t(R_gt[0,:,:].numpy(), t_gt.numpy(), M.detach().cpu().numpy(), M_gt.detach().cpu().numpy())

            num_data += 1
            torch.cuda.empty_cache()

            eval_res["err_q"][batch_idx] = err_q
            eval_res["err_t"][batch_idx] = err_t
            #eval_res["inlier_ratio"][batch_idx] = inlier_ratio

            logging.info(' '.join([
                f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
                f"Matching Time: {matching_timer.avg:.3f},",
                f"err_rot: {err_q:.3f}, err_t: {err_t:.3f}",
            ]))
            data_timer.reset()

        # after checking all the validation samples, let's calculate statistics

        recall = recalls(eval_res)
        logging.info(' '.join([
        f"recall_rot: {recall[0]:.3f}, mean rot. : {recall[1]:.3f}, mean trans. : {recall[2]:.3f}, avg. inlier ratio: {recall[3]:.3f},",
        ]))
        logging.info(' '.join([
            f"med rot. : {recall[4]:.3f}, med trans. : {recall[5]:.3f}",
        ]))
        logging.info(' '.join([
            f"rot Q1. : {recall[6]:.3f}, rot Q3. : {recall[7]:.3f}",
        ]))
        logging.info(' '.join([
            f"trans Q1. : {recall[8]:.3f}, trans Q3. : {recall[9]:.3f}",
        ]))

        total_params = sum(p.numel() for p in model.parameters())
        logging.info(' '.join([f"Total model parameters: {total_params}",]))


if __name__ == '__main__':

    configs = get_config()

    # -------------------------------------------------------------
    """You can change the configurations here or in the file config.py"""

    #configs.dataset = "structured3D"
    configs.dataset = "Apollo"
    configs.data_dir = "./dataset"
    # select which GPU to be used
    configs.gpu_inds = 0
    # This is a model number, set it to whatever you want
    configs.model_nb = "lambda200bs8_larger"
    configs.net_nchannel = 32*6

    configs.weights = os.path.join(configs.out_dir, configs.dataset, configs.model_nb) + '/checkpoint.pth'


    # Convert to dict
    dconfig = vars(configs)
    configs = edict(dconfig)

    main(configs)











