import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import *
from models.networks import *
from models.loss import *
import scipy.io as sio
import sys
from torchinterp1d import Interp1d
import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='experiment',
      help='name for the experiment folder')
    parser.add_argument('--snr', default='high',
      choices=('high', 'low'),
      help='select high SNR scenario or low SNR scenario')

    
    parser.add_argument('--bandwidth', default=40, type=int,
      help='define the system bandwidth')
    parser.add_argument('--e_sr', type=int, default=200,
      help='number of epochs to train the super resolution network')
    parser.add_argument('--lr_sr', type=float, default=1e-3,
      help='learning rate for the super resolution network')
    parser.add_argument('--batch_sr', type=int, default=400,
      help='batchsize for the super resolution network')
    parser.add_argument('--lam', type=float, default=50,
      help='weight for the time domain loss')
    parser.add_argument('--up', type=int, default=2,
      help='upsamping rate for the super resolution net')
    
    parser.add_argument('--e_reg', type=int, default=200,
      help='number of epochs to train the super resolution network')
    parser.add_argument('--lr_reg', type=float, default=1e-3,
      help='learning rate for the super resolution network')
    parser.add_argument('--batch_reg', type=int, default=400,
      help='batchsize for the super resolution network')
    parser.add_argument('--use_ori', type = str2bool, nargs='?',
      default=False,
      help='whether to include the original observation for the regressors')

    parser.add_argument('--window_len', type=float, default=60,
      help='window size for the second regressor')
    parser.add_argument('--window_index', type=int, default=128,
      help='number of samples for the window for the second regressor')

    parser.add_argument('--device', default='cpu',
      choices=('cpu', 'gpu'),
      help='choose the device to run the code')

    parser.add_argument('--print_interval', type=int, default=20,
      help='number of iterations between each loss print')

    return parser

def interp1d(y, x, x_new):
    '''
    linear interpolation for the third dimension
    y: Nx2xL
    x: Nx2xL
    x_new: Nx2xL
    '''
    N = y.shape[0]
    out = []
    for i in range(N):
        y_new = None
        y_new = Interp1d()(x[i], y[i], x_new[i], y_new).unsqueeze(0)   
        out.append(y_new)
            
    return torch.cat(out, 0).detach()              
        

def train_SR(train_loader, G, device, print_interval, lr, N_epochs, lam, step, folder):

    G_solver = optim.Adam(G.parameters(), lr=lr)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(G_solver, step_size=step, gamma=0.3)

    log_path_sr = os.path.join(folder, 'loss_log_sr.txt')
    model_path_sr = os.path.join(folder, 'sr.w')

    for epoch in range(N_epochs):
        
        for i, (frame) in enumerate(train_loader):
            cir_l = frame['cir_l'].float().to(device)
            cir_h = frame['cir_h'].float().to(device)
            cfr_h = frame['cfr_h'].float().to(device)
            dist = frame['dist'].float().to(device)
            
            G_solver.zero_grad()
            cir_h_fake = G(cir_l)
            loss_f = L2_loss_f(cir_h_fake, cfr_h)
            loss_t = L2_loss_t_amp(cir_h_fake, cir_h)
            loss = loss_f + lam*loss_t
            loss.backward()
            G_solver.step()
            
            if (i % print_interval == 0):
                with open(log_path_sr, "a") as log_file:
                    message = '(epoch: %d, iters: %d, loss: %.3f, f-loss: %.3f, t-loss: %.3f) ' % (epoch, i, loss.item(), loss_f.item(), loss_t.item())
                    print(message)  # print the message
                    log_file.write('%s\n' % message)  # save the message

        exp_lr_scheduler.step() 
    
    print('Training sr finished!')
    torch.save(G.state_dict(), model_path_sr)

    return G

def train_Reg(train_loader, G, RA, RB, device, print_interval, win_size, win_idx, lr, N_epochs, step, folder, use_ori=False):

    R_solver = optim.Adam(list(RA.parameters())+list(RB.parameters()), lr=lr)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(R_solver, step_size=step, gamma=0.3)
    criterion = nn.MSELoss()
    
    log_path_R = os.path.join(folder, 'loss_log_reg.txt')

    model_path_RA = os.path.join(folder, 'ra.w')
    model_path_RB = os.path.join(folder, 'rb.w')

    for epoch in range(N_epochs):
        
        for i, (frame) in enumerate(train_loader):
            cir_l = frame['cir_l'].float().to(device)
            cir_h = frame['cir_h'].float().to(device)
            cfr_h = frame['cfr_h'].float().to(device)
            dist = frame['dist'].float().to(device)
            
            R_solver.zero_grad()
            cir_h_fake = G(cir_l)
            if use_ori:
                RA_input = torch.cat((cir_l, cir_h_fake), 1)
            else:
                RA_input = cir_h_fake

            pred_coarse = RA(RA_input)
            
            index = ((pred_coarse > win_size//2) * (pred_coarse < 960-win_size//2)).squeeze()
            x = 3.75 * torch.arange(0, cir_l.shape[2]).repeat(2, 1).to(device)
            window = torch.arange(-win_size, win_size, 2*win_size/win_idx).repeat(2, 1).to(device)
            
            if torch.sum(index) != 0:                
                with torch.no_grad():
                    cir_l_ = cir_l[index]
                    cir_h_fake_ = cir_h_fake[index]
                    pred_coarse_ = pred_coarse[index]
                    x_new = pred_coarse.unsqueeze(-1) + window.unsqueeze(0)
                    x = x.repeat(cir_l.shape[0],1,1)
                    
                    y_new = interp1d(cir_h_fake_, x, x_new) 
                    if use_ori:
                        y_new_ori = interp1d(cir_l_, x, x_new)                   
                        y_new = torch.cat((y_new_ori, y_new), 1).detach()
            
                dist_ = dist[index]
                tar_ = dist_ - pred_coarse_ 
                pred_ = RB(y_new)

                loss_fine = criterion(pred_, tar_)
                loss_coarse = criterion(pred_coarse, dist)
                loss = loss_fine + loss_coarse
                loss.backward()
                R_solver.step()

                if (i % print_interval == 0):
                    with open(log_path_R, "a") as log_file:
                        message = '(epoch: %d, iters: %d, loss_coar: %.3f, loss_fine: %.3f) ' % (epoch, i, loss_coarse.item(), loss_fine.item())
                        print(message)  # print the message
                        log_file.write('%s\n' % message)  # save the message
            else:

                loss_coarse = criterion(pred_coarse, dist)
                loss = loss_coarse
                loss.backward()
                R_solver.step()

                if (i % print_interval == 0):
                    with open(log_path_R, "a") as log_file:
                        message = '(epoch: %d, iters: %d, loss_coar: %.3f) ' % (epoch, i, loss_coarse.item())
                        print(message)  # print the message
                        log_file.write('%s\n' % message)  # save the message

        exp_lr_scheduler.step() 

    print('Training regressors finished!')
    torch.save(RA.state_dict(), model_path_RA)
    torch.save(RB.state_dict(), model_path_RB)


def train(args):
    
    # choose device
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Make the dataset
    dataset_A = dataLoader('data/traindata/Train_x'+str(args.up)+'_'+args.snr+'_'+str(args.bandwidth)+'MHz_A.mat')
    train_loader_A = torch.utils.data.DataLoader(dataset_A, batch_size=args.batch_sr,
                                           shuffle=True, num_workers=1,
                                           pin_memory=True)

    dataset_B = dataLoader('data/traindata/Train_x'+str(args.up)+'_'+args.snr+'_'+str(args.bandwidth)+'MHz_B.mat')
    train_loader_B = torch.utils.data.DataLoader(dataset_B, batch_size=args.batch_reg,
                                           shuffle=True, num_workers=1,
                                           pin_memory=True)

    cir_len = int(args.bandwidth // 0.3125)
    num_channel = 4 if args.use_ori else 2

    # Initialize the models
    G = unet().to(device)
    RA = Regnet(cir_len*args.up, num_channel).to(device)
    RB = Regnet(args.window_index, num_channel).to(device)
    
    # Initialize the log path
    folder = os.path.join('experiments', args.name)
    if os.path.exists(folder) == False:
        os.makedirs(folder)
        
    G_trained = train_SR(train_loader_A, G, device, args.print_interval, args.lr_sr, args.e_sr, args.lam, args.e_sr//4, folder)
    G_trained.eval()
    train_Reg(train_loader_B, G_trained, RA, RB, device, args.print_interval, args.window_len, args.window_index, args.lr_sr, args.e_reg, args.e_reg//4, folder, args.use_ori)



def main():
    parser = make_parser()
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
