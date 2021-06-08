"""
Meta-parametrized SRF layer with alpha and sigma linearly dependent on t.
"""
# Import general dependencies
import numpy as np
import torch
import torch.nn as nn
from srf.gaussian_basis_filters import gaussian_basis_filters_shared
import torch.nn.functional as F

class Srf_layer_alpha_sigma_lin(nn.Module):
    def __init__(self,
                inC,
                outC, 
                init_k,
                init_order,
                init_scale,
                learn_sigma, 
                use_cuda):
        super(Srf_layer_alpha_sigma_lin, self).__init__()

        self.init_k = init_k
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

        self.alphas_bias = torch.nn.Parameter(torch.zeros([F, inC, outC], \
                            device=self.device), requires_grad=True)
        torch.nn.init.normal_(self.alphas_bias, mean=0.0, std=0.05)

        if learn_sigma:    
            self.scales = torch.nn.Parameter(torch.zeros([1],\
                            device=self.device), requires_grad=True)
            torch.nn.init.normal_(self.scales, mean=-0.0, std=2./3.)
            
            self.scale_bias = torch.nn.Parameter(torch.zeros([1],\
                            device=self.device), requires_grad=True)
            torch.nn.init.normal_(self.scale_bias, mean=-0.0, std=1./10.)
        else:
            self.scales = torch.nn.Parameter(torch.tensor(np.full((1), \
                            self.init_scale), device=self.device,\
                            dtype=torch.float32), requires_grad=False)      
        self.extra_reg = 0

    def forward(self, data, t):
        # Define sigma from the scale
        self.sigma = 2.0**(self.scales*t+self.scale_bias)
        filters, _ = gaussian_basis_filters_shared(
                                            order=self.init_order, \
                                            sigma=self.sigma, \
                                            k=self.init_k, \
                                            alphas=self.alphas*t+self.alphas_bias,\
                                            use_cuda=self.use_cuda)
        final_conv = F.conv2d(
                        input=data, # NCHW
                        weight=filters, # KCHW
                        bias=None,
                        stride=1,
                        padding=int(filters.shape[2]/2))

        return final_conv



