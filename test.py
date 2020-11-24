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
import matplotlib.pyplot as plt

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
    parser.add_argument('--snr', default=30, type=int,
      help='select high SNR scenario or low SNR scenario')
    parser.add_argument('--bandwidth', default=40, type=int,
      help='define the system bandwidth')
    parser.add_argument('--up', type=int, default=2,
      help='upsamping rate for the super resolution net')
    parser.add_argument('--use_ori', type = str2bool, nargs='?',
      default=False,
      help='whether to include the original observation for the regressors')
    parser.add_argument('--use_802', type = str2bool, nargs='?',
      default=False,
      help='whether to test the 802.15.4a channel')
    parser.add_argument('--window_len', type=float, default=60,
      help='window size for the second regressor')
    parser.add_argument('--window_index', type=int, default=128,
      help='number of samples for the window for the second regressor')
    parser.add_argument('--device', default='cpu',
      choices=('cpu', 'gpu'),
      help='choose the device to run the code')
    parser.add_argument('--num_test', default=10, type=int,
      help='number of test cirs')
    
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

def plot_cir(cir_l, cir_h, cir_h_fake, pred_coarse, pred_fine, gt, folder_path):

    x = 3.75 * np.arange(0, cir_l.shape[2])
    cir_l = cir_l.detach().cpu().numpy()
    cir_h_fake = cir_h_fake.detach().cpu().numpy()
    pred_coarse = pred_coarse.detach().cpu().numpy()
    pred_fine = pred_fine.detach().cpu().numpy()
    
    for i in range(cir_l.shape[0]):
        
        cir_l_real = cir_l[i,0,:]
        cir_l_imag = cir_l[i,1,:]
        cir_l_amp = np.sqrt(cir_l_real**2 + cir_l_imag**2)
        cir_h_fake_real = cir_h_fake[i,0,:]
        cir_h_fake_imag = cir_h_fake[i,1,:]
        cir_h_fake_amp = np.sqrt(cir_h_fake_real**2 + cir_h_fake_imag**2)
        cir_h_real = cir_h[i,0,:]
        cir_h_imag = cir_h[i,1,:]
        cir_h_amp = np.sqrt(cir_h_real**2 + cir_h_imag**2)
        
        focus = int(pred_fine[i]//3.75) + np.arange(-np.minimum(int(pred_fine[i]//3.75),30), np.minimum(30, cir_l.shape[2]-int(pred_fine[i]//3.75)))
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        line1,= plt.plot(x[focus], cir_l_amp[focus])
        line2,= plt.plot(x[focus], cir_h_amp[focus])
        plt.xlabel('Distance')
        plt.ylabel('Amplitude')
        plt.legend([line1, line2], ['Original noisy low resolution CIR', 'Ground truth high resolution CIR'])
        
        plt.subplot(122)
        line1,= plt.plot(x[focus], cir_h_fake_amp[focus])
        line2,=plt.plot(x[focus], cir_h_amp[focus])
        plt.xlabel('Distance')
        plt.ylabel('Amplitude')
        plt.legend([line1, line2], ['De-noised high resolution CIR', 'Ground truth high resolution CIR'])

        save_path = os.path.join(folder_path, str(i)+'_cir.png')
        plt.savefig(save_path)
        
        plt.figure(figsize=(8, 6))
        line1,= plt.plot(x[focus], cir_h_fake_amp[focus])
        line2,= plt.plot(pred_coarse[i]*np.ones(100), np.arange(0, 1, 0.01), linestyle='--')
        line3,= plt.plot(pred_fine[i]*np.ones(100), np.arange(0, 1, 0.01), linestyle='--')
        line4,= plt.plot(gt[i]*np.ones(100), np.arange(0, 1, 0.01), linestyle='--', color='black')
        plt.xlabel('Distance')
        plt.ylabel('Amplitude')
        plt.legend([line1, line2, line3, line4], ['De-noised high resolution CIR', 'coarse estimation', 'fine estimation', 'ground truth'])
        save_path = os.path.join(folder_path, str(i)+'_toa.png')
        plt.savefig(save_path)


def test(args):
    
    # choose device
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.use_802:
        mat_data = sio.loadmat('data/testdata/Test_x'+str(args.up)+'_'+str(args.snr)+'dB_'+str(args.bandwidth)+'MHz_802.mat')
    else:
        mat_data = sio.loadmat('data/testdata/Test_x'+str(args.up)+'_'+str(args.snr)+'dB_'+str(args.bandwidth)+'MHz.mat')

    cir_l= mat_data['cir_l']
    cir_h= mat_data['cir_h']
    dist = torch.from_numpy(mat_data['dist']).to(device)

    cir_len = int(args.bandwidth // 0.3125)
    num_channel = 4 if args.use_ori else 2
    
    folder = os.path.join('experiments', args.name)

    # Initialize the models
    G = unet().to(device)
    RA = Regnet(cir_len*args.up, num_channel).to(device)
    RB = Regnet(args.window_index, num_channel).to(device)
    
    G.load_state_dict(torch.load(os.path.join(folder, 'sr.w')))
    RA.load_state_dict(torch.load(os.path.join(folder, 'ra.w')))
    RB.load_state_dict(torch.load(os.path.join(folder, 'rb.w')))
    
    G.eval()
    RA.eval()
    RB.eval()
    
    cir_l_test = cir_l[:args.num_test]
    cir_l_test_torch = torch.from_numpy(cir_l_test).float().to(device)
    if cir_l_test_torch.dim() == 1:
        cir_l_test_torch = cir_l_test_torch.unsqueeze(0)

    cir_h_fake = G(cir_l_test_torch)

    if args.use_ori:
        RA_input = torch.cat((cir_l_test_torch, cir_h_fake), 1)
    else:
        RA_input = cir_h_fake

    pred_coarse = RA(RA_input)

    x = 3.75 * torch.arange(0, cir_l.shape[2]).repeat(args.num_test, 2, 1).to(device)
    window = torch.arange(-args.window_len, args.window_len, 2*args.window_len/args.window_index).repeat(1, 2, 1).to(device)
    x_new = pred_coarse.unsqueeze(-1) + window
    
    with torch.no_grad():
        y_new = interp1d(cir_h_fake, x, x_new) 
        if args.use_ori:
            y_new_ori = interp1d(cir_l_test_torch, x, x_new)                   
            y_new = torch.cat((y_new_ori, y_new), 1).detach()
    
    pred_fine = RB(y_new)

    pred_final = pred_coarse+pred_fine
    
    rmse_coarse = torch.sqrt(torch.mean((dist[:args.num_test]-pred_coarse)**2)).item()
    rmse_final = torch.sqrt(torch.mean((dist[:args.num_test]-pred_final)**2)).item()
    print('RMSE (coarse): %.3f, RMSE (fine): %.3f' % (rmse_coarse, rmse_final))
    
    plot_folder = os.path.join(folder, 'figures_'+str(args.snr)+'_dB')
    if os.path.exists(plot_folder) == False:
        os.makedirs(plot_folder)

    plot_cir(cir_l_test_torch, cir_h, cir_h_fake, pred_coarse, pred_final, dist[:args.num_test], plot_folder)
    



def main():
    parser = make_parser()
    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
