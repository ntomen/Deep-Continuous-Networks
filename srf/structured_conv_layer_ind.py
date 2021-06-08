"""
SRF layer where each filter is fit its own sigma independently. (inC x outC sigmas are trained per layer.)
"""

# Import general dependencies
import numpy as np
import torch
import torch.nn as nn
from srf.gaussian_basis_filters import gaussian_basis_filters_ind
import torch.nn.functional as F

class Srf_layer_ind(nn.Module):
    def __init__(self,
                inC,
                outC, 
                k_size,
                init_order,
                init_scale,
                learn_sigma, 
                use_cuda):
        super(Srf_layer_ind, self).__init__()

        self.k_size = k_size 
        self.init_order = init_order
        self.init_scale = init_scale
        self.inC = inC
        self.outC = outC
        #---------------
        F = int((self.init_order + 1) * (self.init_order + 2) / 2)                        

        """ Create weight variables. """
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.alphas = torch.nn.Parameter(torch.zeros([F, inC, outC], \
                            device=self.device), requires_grad=True) 
        torch.nn.init.normal_(self.alphas, mean=0.0, std=0.1)
        if learn_sigma:    
            self.scales = torch.nn.Parameter(torch.zeros([inC, outC], \
                            device=self.device), requires_grad=True)      
            torch.nn.init.normal_(self.scales, mean=0.0, std=2./3.)
        else:
            self.scales = torch.nn.Parameter(torch.tensor(np.full((inC, outC), \
                            self.init_scale), device=self.device,\
                            dtype=torch.float32), requires_grad=False)      

    def forward(self, data, renew_filters):
#        if renew_filters:
        # Define sigma from the scale
        self.sigma = 2.0**self.scales
        self.filters, _ = gaussian_basis_filters_ind(
                                            order=self.init_order, \
                                            sigma=self.sigma.reshape(self.inC*self.outC), \
                                            k_size=self.k_size, \
                                            alphas=self.alphas,\
                                            use_cuda=self.use_cuda)

        final_conv = F.conv2d(
                        input=data, # NCHW
                        weight=self.filters, # KCHW
                        bias=None,
                        stride=1,
                        padding=int(self.filters.shape[2]/2))

        return final_conv












