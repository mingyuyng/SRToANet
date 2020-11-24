import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def MSE_loss(input, target):

    loss = None
    criterion = nn.MSELoss()
    loss = criterion(input, target)
    return loss


def L2_loss_f(input, target):
    '''
    Compute the L2 loss for channel frequency response

    Inputs:
    - input:  PyTorch Variable of shape (N, 2, L), which is the estimated CIR
    - target: PyTorch Variable of shape (N, 2, L), which is the target CFR

    Returns:
    - loss: PyTorch Variable containing the (scalar) L2 loss
    '''
      
    input_f = torch.fft(input.transpose(1, 2), 1)    # N x L x 2,  batched 1D FFT
    input_f = input_f.transpose(1, 2)                # N x 2 x L,  batched 1D FFT

    criterion = nn.MSELoss()
    loss = criterion(input_f, target)

    return loss


def L2_loss_t_amp(input, target):
    '''
    Compute the L2 loss in the time domain

    Inputs:
    - gen_out: PyTorch Variable of shape (N, 2, L), which is the output of generator
    - target: PyTorch Variable of shape (N, 2, L), which is the ground truth

    Returns:
    - loss: PyTorch Variable containing the (scalar) L2 loss
    '''
    input_amp = torch.sum(input**2, 1)
    target_amp = torch.sum(target**2, 1)
    criterion = nn.MSELoss()
    loss = criterion(target_amp, input_amp)

    return loss