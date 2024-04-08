# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import copy
from collections import OrderedDict

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--adv_attacker', default=1, type=int,
                        help='whether to do an adversarial attack ')
    parser.add_argument('--adv_eps', default=0.01, type=float,
                        help='magnitude of adversarial perturbation ')
    parser.add_argument('--adv_alpha', default=0.1, type=float,
                        help='alpha of adversarial perturbation ')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)
    # print(os.environ)
    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    unperturbed_model = copy.deepcopy(model)
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):

            #if there is more than one CAV in the scene
            if 1 < batch_data['ego']['record_len'][0]:
                # the model will be evaluation mode during validation
                model.train()
                model.zero_grad()
                optimizer.zero_grad()

                batch_data = train_utils.to_device(batch_data, device)

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

                # adversarial attack steps (PGD)
                adv_steps = 5
                for _ in range(adv_steps):
                    pert.requires_grad = True
                    # Introduce adv perturbation
                    batch_data['ego']['pert'] = pert.to(device)
                    output = model.adv_step(batch_data['ego'], attacker, adv_eps)
                    # NOTE: Actual ground truth is not always available especially in real-world attacks
                    # We define the adversarial loss of the perturbed output
                    # with respect to an unperturbed output (pseudo_dict) instead of the ground truth
                    loss = criterion(output, pseudo_dict)
                    #loss *= -1
                    batch_len = len(train_loader)
                    writer.add_scalar('Regression_loss', loss.item(),
                                      epoch * batch_len + i)
                    grad = torch.autograd.grad(loss, pert, retain_graph=False, create_graph=False)[0]
                    pert = pert + pert_alpha * grad.sign()
                    pert.detach_()

                # Detach and clone perturbations from Pytorch computation graph, in case of gradient misuse.
                pert = pert.detach().clone()

                # Apply the final perturbation to attackers' feature maps.
                batch_data['ego']['pert'] = pert.to(device)


                #continue normal/adversarial training
                for k, v in model.named_parameters():
                    v.requires_grad = True  # update parameters for adv training forward

                if not opt.half:
                    #ouput_dict = model(batch_data['ego'])
                    output_dict = model.adv_step(batch_data['ego'], attacker, adv_eps)
                    # first argument is always your output dictionary,
                    # second argument is always your label dictionary.
                    final_loss = criterion(output_dict,
                                           batch_data['ego']['label_dict'])
                else:
                    with torch.cuda.amp.autocast():
                        #ouput_dict = model(batch_data['ego'])
                        output_dict = model.adv_step(batch_data['ego'], attacker, adv_eps)
                        final_loss = criterion(output_dict,
                                               batch_data['ego']['label_dict'])


                criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
                pbar2.update(1)
                #
                if not opt.half:
                    final_loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(final_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()


            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'adv_net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
