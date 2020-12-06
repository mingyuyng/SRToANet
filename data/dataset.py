import torch.utils.data as data
import scipy.io as sio
import numpy as np
import os
import glob



class dataLoader(data.Dataset):

    def __init__(self, root):
        
        mat_data = sio.loadmat(root)
        self.cir_l = mat_data['cir_l'].astype('double')
        self.cir_h = mat_data['cir_h'].astype('double')
        self.cfr_h = mat_data['cfr_h'].astype('double')
        self.dist = mat_data['dist'].astype('double')

    def __getitem__(self, index):

        cir_l = self.cir_l[index]
        cir_h = self.cir_h[index]
        cfr_h = self.cfr_h[index]
        dist = self.dist[index]

        pair = {'cir_l': cir_l, 'cir_h': cir_h,'cfr_h': cfr_h, 'dist': dist}
        return pair

    def __len__(self):
        return self.dist.shape[0]
