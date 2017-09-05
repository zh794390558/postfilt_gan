from __future__ import print_function
import argparse
import os
import random
import shutil
#import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from data_loader import get_loader
from utils import plot_feats, read_binary_file
from models import define_netD, define_netG


def train(netD, netG, data_loader, opt):
    label = torch.FloatTensor(1)
    label = Variable(label, requires_grad=False)
    real_label = 1
    fake_label = 0

    # cost criterion
    #criterion = nn.BCELoss() # normal gan
    criterion = nn.MSELoss() # lsgan

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        criterion.cuda()

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    print('Epoch is {}:{}'.format(int(opt.epochStart), opt.niter))
    for epoch in range(int(opt.epochStart), opt.niter):
        for i, data in enumerate(data_loader, 0):
            # discard the data if the size less than opt.mgcDim frames
            if data[0].size(-1) <= opt.mgcDim:
                continue
            real_data, pred_data = data # NAT, SYN

            #################################
            # (1) Updata D network: maximize log(D(x)) + log(1 - D(G(z)))
            #################################
            # clear the gradient buffers
            netD.zero_grad()
            # train with real
            # crop the tensor to fixed size
            rand_int = random.randint(0, real_data.size(-1) - opt.mgcDim)
            real_data_crop = real_data[:,:,:,rand_int:rand_int+opt.mgcDim]
            label.data.fill_(real_label)

            if opt.cuda:
                pred_data = pred_data.cuda()
                label = label.cuda()
                real_data_crop = real_data_crop.cuda()

            pred_data = Variable(pred_data, requires_grad=False)
            real_data_crop = Variable(real_data_crop, requires_grad=False)

            output = netD(real_data_crop)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            noise = torch.FloatTensor(real_data.size()).normal_(0,1)
            if opt.cuda:
                noise = noise.cuda()
            noise = Variable(noise, requires_grad=False)
            fake = netG(noise, pred_data)
            # add the residual to the tts predicted data
            fake = fake + pred_data
            label.data.fill_(fake_label)
            # crop the tensor to fixed size
            fake_crop = fake[:,:,:,rand_int:rand_int+opt.mgcDim]
            output = netD(fake_crop.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real.data[0] + errD_fake.data[0]
            # update the discriminator on mini batch
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################
            netG.zero_grad()
            label.data.fill_(real_label) # fake labels are real for generator cost
            output = netD(fake_crop)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            if i % opt.logInterval == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    %(epoch, opt.niter, i, len(data_loader), errD, errG.data[0], D_x, D_G_z1, D_G_z2))

                # G
                fake = netG(noise, pred_data)
                fake = fake + pred_data
                fake = fake.data.cpu().numpy()
                fake = fake.reshape(opt.mgcDim, -1)
                fake = fake[:,rand_int:rand_int+opt.mgcDim]

                # SYN
                pred = pred_data.data.cpu().numpy()
                pred = pred.reshape(opt.mgcDim, -1)
                pred = pred[:,rand_int:rand_int+opt.mgcDim]

                # NAT
                real = real_data.cpu().numpy()
                real = real.reshape(opt.mgcDim, -1)
                real = real[:,rand_int:rand_int+opt.mgcDim]

                plot_feats(real, pred, fake,  epoch, i, opt.outf)

            #del errD_fake, errD_real, errG, real_data, pred_data,
            #del noise, real_data_crop, fake, fake_crop, output, errD

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' %(opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' %(opt.outf, epoch))

def test(netG, opt):
    assert opt.netG != ''
    print('mode: {}'.format(opt.mode))

    test_dir = opt.testdata_dir
    output_dir = os.path.join(test_dir, 'out')
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    for f in os.listdir(test_dir):
        fname, ext = os.path.splitext(f)
        if ext == '.mcep':
            print(fname + ext)
            cmp_file = os.path.join(test_dir, f)

            ac_data, nframe = read_binary_file(cmp_file, dim=41)
            ac_data = torch.FloatTensor(ac_data)
            print(ac_data.size())
            noise = torch.FloatTensor(ac_data.size())
            if opt.cuda:
                netG.cuda()
                ac_data, noise = ac_data.cuda(), noise.cuda()
            ac_data = Variable(ac_data)
            noise = Variable(noise)
            noise.data.normal_(0, 1)

            generated_pulses = netG(noise, ac_data)
            generated_pulses = generated_pulses.data.cpu().numpy()
            generated_pulses = generated_pulses.reshape(ac_data.size(0), -1)

            out_file = os.path.join(output_dir, fname + ext)
            with open(out_file, 'wb') as fid:
                generated_pulses.tofile(fid)


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--voiceName', required=True, help='nick | jenny ')
    parser.add_argument('--mode', required=True, type=str, help='train | test')
    parser.add_argument('--xFilesList', required=True, help='path to input files list')
    parser.add_argument('--yFilesList', required=True, help='path to output files list')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--mgcDim', type=int, default=60, help='mel-cepstrum dimension')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--testdata_dir', type=str, help='path to test data')
    parser.add_argument('--logInterval', type=int, default=200, help='log interval')
    opt = parser.parse_args()
    print(opt)


  # prepare the data loader
    x_files_list_file = opt.xFilesList
    y_files_list_file = opt.yFilesList
    in_dim = opt.mgcDim
    out_dim = opt.mgcDim

    with open(x_files_list_file, 'r') as fid:
        x_files_list = [l.strip() for l in fid.readlines()]

    with open(y_files_list_file, 'r') as fid:
        y_files_list = [l.strip() for l in fid.readlines()]


    data_loader = get_loader(x_files_list, y_files_list,
                            in_dim, out_dim, 1, False, opt.workers)

    # prepare the output directories
    try:
        os.makedirs(opt.outf)
        os.makedirs(os.path.join(opt.outf, 'figures'))
    except OSError:
        pass

    # if manual seed is not provide then pick one randomly
    if opt.manualSeed is None:
        opt.manualSeed  = random.randint(1, 10000)
    print('Random Seed: ', opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.enabled = True
    cudnn.benchmark = True

    # epoch is start
    opt.epochStart = os.path.splitext(opt.netG)[0]
    opt.epochStart = opt.epochStart.split('_')[-1]

    # define the generator
    netG = define_netG(in_ch=2)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
        print('Use chekpoint:{}'.format(opt.netG))
    print(netG)

    # define the discriminator
    netD = define_netD(opt.ngpu)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
        print('Use chekpoint:{} device: {}'.format(opt.netD, netD.device))
    print(netD)

    if opt.mode == 'train':
        train(netD, netG, data_loader, opt)
    elif opt.mode == 'test':
        test(netG, opt)
    else:
        print('Mode must be either train or test only')
