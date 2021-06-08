"""
Basic SRF layer which uses Gaussian SRF filters instead of pixel-based filters. 
"""

# Import general dependencies
import numpy as np
import torch
import torch.nn as nn
from srf.gaussian_basis_filters import gaussian_basis_filters, \
                                       gaussian_basis_filters_shared
import torch.nn.functional as F
import time

class Srf_layer(nn.Module):
    def __init__(self,
                inC,
                outC, 
                num_scales,
                init_k,
                init_order,
                init_scale,
                learn_sigma, 
                use_cuda):

        super(Srf_layer, self).__init__()
        self.init_k = init_k
        self.init_order = init_order
        self.init_scale = init_scale
        self.in_channels = inC
        self.out_channels = outC
        #---------------
        F = int((self.init_order + 1) * (self.init_order + 2) / 2)                        

        """ Create weight variables. """
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.scales = nn.ParameterList([])

        # Init alphas
        self.alphas = torch.nn.Parameter(torch.zeros([F, inC, outC], \
                            device=self.device), requires_grad=True) 
        torch.nn.init.normal_(self.alphas, mean=0.0, std=0.1)

        scales_init = np.random.normal(loc=0.0, scale=2./3., size=num_scales)
        for i in range(0, num_scales):
            if learn_sigma:
                self.scales.append(torch.nn.Parameter(torch.tensor(\
                        scales_init[i], device=self.device), requires_grad=True))
            else:
                self.scales.append(torch.nn.Parameter(torch.tensor(\
                        self.init_scale, device=self.device, dtype=torch.float32), \
                        requires_grad=False))
        self.extra_reg = 0
    
    def forward_no_input(self): 
        t = time.time()
        filters = []
        for i in range(0, len(self.scales)):
            sigma = 2.0**self.scales[i]
            print("current sigma: ", sigma)

            one_filter, _ = gaussian_basis_filters(
                                            order=self.init_order, \
                                            sigma=sigma, \
                                            k=self.init_k, \
                                            alphas=self.alphas,\
                                            use_cuda=self.use_cuda)
            filters.append(one_filter)
        
        elapsed = time.time()
        print("Forward pass timing ", elapsed-t)
        return filters

    def forward(self, indata): 
        self.filters = []
        conv_output = []
        conv_output = torch.zeros([indata.shape[0], self.out_channels, \
                            indata.shape[2], indata.shape[3]], \
                            device=self.device)

        for i in range(0, len(self.scales)):
            sigma = 2.0**self.scales[i]
            one_filter, _ = gaussian_basis_filters(
                                        order=self.init_order, \
                                        sigma=sigma, \
                                        k=self.init_k, \
                                        alphas=self.alphas,\
                                        use_cuda=self.use_cuda)


            conv_output += F.conv2d(
                                input=indata,
                                weight=one_filter, # KCHW
                                bias=None,
                                stride=1,
                                padding=int(one_filter.shape[2]/2))

        return conv_output

    def listParams(self):
        params = list(self.parameters())
        total_params = 0 

        for i in range(0, len(params)):
            total_params  = total_params + np.prod(list(params[i].size()))

        print('Total parameters: ', total_params)


class Srf_layer_shared(Srf_layer):
    def __init__(self,
                inC,
                outC, 
                init_k,
                init_order,
                init_scale,
                learn_sigma, 
                use_cuda,
                stride=1):
        super(Srf_layer, self).__init__()

        self.init_k = init_k
        self.init_order = init_order
        self.init_scale = init_scale
        self.inC = inC
        self.outC = outC
        self.stride = stride
        #---------------
        F = int((self.init_order + 1) * (self.init_order + 2) / 2)                        

        """ Create weight variables. """
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.alphas = torch.nn.Parameter(torch.zeros([F, inC, outC], \
                            device=self.device), requires_grad=True) 

        torch.nn.init.normal_(self.alphas, mean=0.0, std=0.1)
        if learn_sigma:    
            self.scales = torch.nn.Parameter(torch.zeros([1], \
                            device=self.device), requires_grad=True)      
            torch.nn.init.normal_(self.scales, mean=0.0, std=2./3.)
        else:
            self.scales = torch.nn.Parameter(torch.tensor(np.full((1), \
                            self.init_scale), device=self.device,\
                            dtype=torch.float32), requires_grad=False)      
        self.extra_reg = 0

    def forward_no_input(self):
        # Define sigma from the scale
        self.sigma = 2.0**self.scales

        filters, _ = gaussian_basis_filters_shared(
                                            order=self.init_order, \
                                            sigma=self.sigma, \
                                            k=self.init_k, \
                                            alphas=self.alphas,\
                                            use_cuda=self.use_cuda)
        return filters


    def forward(self, data): 
        # Define sigma from the scale
        self.sigma = 2.0**self.scales

        filters, _ = gaussian_basis_filters_shared(
                                            order=self.init_order, \
                                            sigma=self.sigma, \
                                            k=self.init_k, \
                                            alphas=self.alphas,\
                                            use_cuda=self.use_cuda)
        final_conv = F.conv2d(
                        input=data, # NCHW
                        weight=filters, # KCHW
                        bias=None,
                        stride=self.stride,
                        padding=int(filters.shape[2]/2))

        return final_conv

