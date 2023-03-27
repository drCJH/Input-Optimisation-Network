import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import math as maths
from torchvision import transforms
import network
import ION_unet


class Net(nn.Module):

    def __init__(self, arch, o, input_sizes=None, output_sizes=None):
        super(Net, self).__init__()

        o.log("building net: " + arch)

        if arch.lower() == "deeplabv3":
            self.model = network.deeplabv3plus_mobilenet(num_classes=19, output_stride=16, pretrained_backbone=False)
        elif "unet" in arch.lower():
            self.model = ION_unet.unet(arch, input_sizes, output_sizes)
        else:
            o.log("invalid architecture: '" + arch + "'")
            exit()

        o.log("model contains " + str(sum(p.numel() for p in self.model.parameters())) + " parameters")
        
        self._initialize_weights()


    def forward(self, x):
        #self.zerograd()
        y = self.model.forward(x)

        y[0] = y[0] - y[0].mean()
        y[0] = y[0] / y[0].std()

        return y

    def parameters(self):
        return self.model.parameters()

    def zerograd(self):
        for param in self.model.model:
            param.grad = None

    def _initialize_weights(self):
        for m in self.modules():                    
            if isinstance(m, nn.Conv2d):                
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)    







        

