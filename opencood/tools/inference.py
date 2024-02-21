# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm
import copy
from collections import OrderedDict

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--adv_attack', action='store_true',
                        help='whether to do an adversarial attack ')
    parser.add_argument('--adv_attacker', default=1, type=int,
                        help='whether to do an adversarial attack ')
    parser.add_argument('--adv_eps', default=0.01, type=float,
                        help='magnitude of adversarial perturbation ')
    parser.add_argument('--adv_alpha', default=0.1, type=float,
                        help='alpha of adversarial perturbation ')
    parser.add_argument('--random_perturb', action='store_true',
                        help='whether to do perturb the intermediate features of an attacker randomly ')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)


    #for k, v in model.named_parameters():
        #print(v.requires_grad)

    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                   0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())
    unperturbed_model = copy.deepcopy(model)
    for i, batch_data in tqdm(enumerate(data_loader)):
        # print(i)
        batch_data = train_utils.to_device(batch_data, device)
        ###adversarial attack#####
        if opt.adv_attack and 1 < batch_data['ego']['record_len'][0]:

            criterion = train_utils.create_loss(hypes)
            model.train()
            # freeze model parameters
            for k, v in model.named_parameters():
                v.requires_grad = False

            # fix attackers, agent 1 always attacks
            # could sample a random attacker as well, or be learned
            attacker = opt.adv_attacker

            # initialize perturbation magnitude
            adv_eps = opt.adv_eps
            # batch_data['adv_eps'] = adv_eps

            # initialize perturbation alpha
            pert_alpha = opt.adv_alpha

            # initialize perturbation tensor
            pert = torch.empty(384, 96, 352).uniform_(-adv_eps, adv_eps)

            #get output of model without an attack
            output_ = unperturbed_model(batch_data['ego'])
            output_['rm'] = output_['rm'].permute(0,2,3,1)
            output_['psm'] = output_['psm'].permute(0,2,3,1)
            output_['psm'] = torch.sigmoid(output_['psm'])
            output_['psm'][output_['psm'] < 0.5] = 0
            output_['psm'][output_['psm'] >= 0.5] = 1
            pseudo_dict = OrderedDict()
            pseudo_dict['targets'] = output_['rm']
            pseudo_dict['pos_equal_one'] = output_['psm']
            #log_probs_ = torch.nn.functional.softmax(cls_preds, dim=1)


            # adversarial attack steps (PGD)
            adv_steps = 10
            for _ in range(adv_steps):
                pert.requires_grad = True
                # Introduce adv perturbation
                batch_data['ego']['pert'] = pert.to(device)
                output = model.adv_step(batch_data['ego'], attacker, adv_eps)
                #psm = output['psm']
                #cls_preds = psm.permute(0, 2, 3, 1).contiguous()
                #cls_preds = cls_preds.view(psm.shape[0], -1, 1)
                #log_probs = torch.nn.functional.softmax(cls_preds, dim=1)


                #print(output['psm'].shape)
                #loss = criterion(output, batch_data['ego']['label_dict'])
                # NOTE: Actual ground truth is not always available especially in real-world attacks
                # We define the adversarial loss of the perturbed output
                # with respect to an unperturbed output (pseudo_dict) instead of the ground truth
                #loss = torch.nn.functional.kl_div(log_probs_, log_probs, reduction='batchmean')
                loss = criterion(output, pseudo_dict)
                #loss *= -1
                grad = torch.autograd.grad(loss, pert, retain_graph=False, create_graph=False)[0]
                pert = pert + pert_alpha * grad.sign()
                pert.detach_()

            # Detach and clone perturbations from Pytorch computation graph, in case of gradient misuse.
            pert = pert.detach().clone()

            # Apply the final perturbation to attackers' feature maps.
            batch_data['ego']['pert'] = pert.to(device)

        model.eval()
        with torch.no_grad():
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset,
                                                           opt)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset,
                                                                  opt)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
