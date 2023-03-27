import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import math as maths
from torchvision import transforms
import network


class unet(nn.Module):

    def __init__(self, arch, input_sizes, output_sizes):
        super(Net, self).__init__()


        if arch.lower() == "unet":
            
            i_sz0 = 384
            i_sz = i_sz0
            
            config = [('C', 64), ('B'), ('R'), ('C'), ('B'), ('R')]
            routing = [('i0',), (-1,), (-1,), (-1,), (-1,), (-1,)]

            num_feats = 64
            feat_inc = num_feats/2

            #keep adding encoder layers until input is smaller than convolution kernel
            while i_sz > 3:
                #change feat_inc if num_feats is a power of 2
                if maths.log2(num_feats) == int(maths.log2(num_feats)):
                    feat_inc = int(num_feats / 2)
                num_feats += feat_inc

                config += [('P'), ('C', num_feats), ('B'), ('R'), ('C'), ('B'), ('R')]
                routing += [(-1,), (-1,), (-1,), (-1,), (-1,), (-1,), (-1,)]

                i_sz /= 2
                


            #build equal length decoder with skip connections
            skip_con = -1
            #while i_sz < min(input_sizes[0][1], input_sizes[0][2]):
            while i_sz < i_sz0:
                #change feat_inc if num_feats is a power of 2
                num_feats -= feat_inc
                if maths.log2(num_feats) == int(maths.log2(num_feats)):
                    feat_inc = int(num_feats / 4)

                config += [('T'), ('C', num_feats), ('B'), ('R'), ('C'), ('B'), ('R')]                
                routing += [(-1,), (-1, (skip_con * 7) - 2), (-1,), (-1,), (-1,), (-1,), (-1,)]

                i_sz *= 2                
                skip_con -= 2
               


            config += [('C', output_sizes[0][0], 1), ('B',), ('H',)]
            routing += [(-1, 'i0'), (-1,), (-1,)]

        elif arch.lower() == "unets":
            
            #i_sz = min(input_sizes[0][1], input_sizes[0][2])
            i_sz0 = 384
            i_sz = i_sz0
            
            #in convolution
            config = [('C', 64), ('B'), ('R'), ('C'), ('B'), ('R')]
            routing = [('i0',), (-1,), (-1,), (-1,), (-1,), (-1,)]

            num_feats = 64
            feat_inc = num_feats/2

            #keep adding encoder layers until input is smaller than convolution kernel
            while i_sz > 16:
                #change feat_inc if num_feats is a power of 2
                if maths.log2(num_feats) == int(maths.log2(num_feats)):
                    feat_inc = int(num_feats / 2)
                num_feats += feat_inc

                config += [('P'), ('C', num_feats), ('B'), ('R'), ('C'), ('B'), ('R')]
                routing += [(-1,), (-1,), (-1,), (-1,), (-1,), (-1,), (-1,)]

                i_sz /= 2
                


            #build equal length decoder with skip connections
            skip_con = -1
            #while i_sz < min(input_sizes[0][1], input_sizes[0][2]):
            while i_sz < i_sz0:
                #change feat_inc if num_feats is a power of 2
                num_feats -= feat_inc
                if maths.log2(num_feats) == int(maths.log2(num_feats)):
                    feat_inc = int(num_feats / 4)

                config += [('T'), ('C', num_feats), ('B'), ('R'), ('C'), ('B'), ('R')]                
                routing += [(-1,), (-1, (skip_con * 7) - 2), (-1,), (-1,), (-1,), (-1,), (-1,)]

                i_sz *= 2                
                skip_con -= 2
               


            config += [('C', output_sizes[0][0], 1), ('B',), ('H',)]
            routing += [(-1, 'i0'), (-1,), (-1,)]

        elif arch.lower() == "unetss":
            
            #i_sz = min(input_sizes[0][1], input_sizes[0][2])
            i_sz0 = 384
            i_sz = i_sz0
            
            #in convolution
            config = [('C', 64), ('B'), ('R'), ('C'), ('B'), ('R')]
            routing = [('i0',), (-1,), (-1,), (-1,), (-1,), (-1,)]

            num_feats = 64
            feat_inc = num_feats/2

            #keep adding encoder layers until input is smaller than convolution kernel
            while i_sz > 64:

                #change feat_inc if num_feats is a power of 2
                if maths.log2(num_feats) == int(maths.log2(num_feats)):
                    feat_inc = int(num_feats / 2)
                num_feats += feat_inc

                config += [('P'), ('C', num_feats), ('B'), ('R'), ('C'), ('B'), ('R')]
                routing += [(-1,), (-1,), (-1,), (-1,), (-1,), (-1,), (-1,)]

                i_sz /= 2
                


            #build equal length decoder with skip connections
            skip_con = -1
            #while i_sz < min(input_sizes[0][1], input_sizes[0][2]):
            while i_sz < i_sz0:
                #change feat_inc if num_feats is a power of 2
                num_feats -= feat_inc
                if maths.log2(num_feats) == int(maths.log2(num_feats)):
                    feat_inc = int(num_feats / 4)

                config += [('T'), ('C', num_feats), ('B'), ('R'), ('C'), ('B'), ('R')]                
                routing += [(-1,), (-1, (skip_con * 7) - 2), (-1,), (-1,), (-1,), (-1,), (-1,)]

                i_sz *= 2                
                skip_con -= 2
               


            config += [('C', output_sizes[0][0], 1), ('B',), ('H',)]
            routing += [(-1, 'i0'), (-1,), (-1,)]

        self.model = NetBuilder(config, input_sizes, routing=routing)




    def forward(self, x):
        #self.zerograd()
        y = self.model.forward(x)
        return y

    def parameters(self):
        return self.model.parameters()

    def zerograd(self):
        for param in self.model.model:
            param.grad = None






def buildconv(l, in_sz, bias=True):        

    out_ch = in_sz[0]
    ksize = 3
    stride = 1
    pad = int(ksize/2)

    if len(l) > 1:
        out_ch = l[1]

    if len(l) > 2:
        ksize = l[2]
        pad = int(ksize/2)

    if len(l) > 3:
        stride = l[3]
    
    if len(l) > 4:
        pad = l[4]


    layer = nn.Conv2d(in_sz[0], out_ch, ksize, stride, pad, bias=bias)     
    test_in = torch.zeros([1] + in_sz.tolist())
    test_out = layer(test_in)

    return layer, test_out.shape[1:]


def buildlinear(l, in_sz):    

    out_sz = in_sz
    bias = 1

    if len(l) > 1:
        out_sz = l[1]

    if len(l) > 2:
        bias = l[2]
    
    return nn.Linear(in_sz, out_sz, bias=bias), [out_sz, 1, 1]


def buildpool(l, in_sz):    

    ksize = 2
    stride = 2
    pad = 0

    if len(l) > 1:
        ksize = l[1]

    if len(l) > 2:
        stride = l[2]

    if len(l) > 3:
        pad = l[3]


    layer = nn.MaxPool2d(ksize, stride=stride, padding=pad)
    
    test_in = torch.zeros([1] + in_sz.tolist())
    test_out = layer(test_in)

    return layer, test_out.shape[1:]

def buildtranspose(l, in_sz, out_sz=[]):
    """
    out_ch = in_sz[0]
    ksize = 2
    stride = 2
    pad = 0

    if len(l) > 1:
        out_ch = l[1]

    if len(l) > 2:
        ksize = l[2]
        pad = int(ksize/2)

    if len(l) > 3:
        stride = l[3]
    
    if len(l) > 4:
        pad = l[4]    
    """
    # Transpose conv swapped for bicubic upsampling

    if len(out_sz):
        layer = nn.Upsample(size=out_sz, mode='bicubic')        
    else:        
        layer = nn.Upsample(scale_factor=2, mode='bicubic')    

    test_in = torch.zeros([1] + in_sz.tolist())
    test_out = layer(test_in)

    return layer, test_out.shape[1:]



class NetBuilder(nn.Module):
    
    def __init__(self, layers, model_input_sizes, model_outputs=[-1], routing=[("i0",)]):
        super(NetBuilder, self).__init__()

        
        #layer indices for input to each layer, ie -1 means output of previous layer
        for i in range(len(routing)):
            #check each element is a tuple
            if type(routing[i]) != type((0,)):                           
                if type(routing[i]) == type([]):
                    routing[i] = tuple(routing[i])
                else:
                    routing[i] = (routing[i],)

                
            
        if len(routing) < len(layers):                        
            #ensure routing is present for every layer
            routing += [(-1,)] * (len(layers) - len(routing)) 

        
        self.config = layers
        self.routing = routing        
        self.model = nn.ModuleList()
        self.firstrun = True
        self.single_input = False
        self.model_outputs = model_outputs

        input_dimensions = 3

        layer_output_sizes = np.zeros((len(layers), input_dimensions), dtype=np.int)
        layer_input_sizes = np.zeros((len(layers), input_dimensions), dtype=np.int)

        

        for i in range(len(layers)):             
            for r in routing[i]:                
                if 'i' in str(r):              
                    #layer takes model input as input
                    try:
                        if layer_input_sizes[i,0] == 0:
                            layer_input_sizes[i,:] = model_input_sizes[int(r[1:])]                       
                        else:
                            layer_input_sizes[i,0] += model_input_sizes[int(r[1:])][0]  
                    except:
                        if layer_input_sizes[i,0] == 0:
                            layer_input_sizes[i,:] = model_input_sizes[0]
                        else:
                            layer_input_sizes[i,0] += model_input_sizes[0][0]
                        r = "i0"                       
                else:
                    try:
                        #cast as int just in case 
                        r = int(r)

                        if r < 0:
                            r = i + r
                        
                        if layer_input_sizes[i,0] == 0:
                            layer_input_sizes[i,:] = layer_output_sizes[r,:]
                        else:
                            layer_input_sizes[i,0] += layer_output_sizes[r,0]
                            
                    except:
                        o.log("incorrect argument for routing: '" + r + "' in '" + str(routing[i]) + "'")                       
                        exit()
            
            #try:
            if layers[i][0].upper() == 'C':
                layer, out_size = buildconv(layers[i], layer_input_sizes[i,:], bias=False)
                self.model.append(layer)                
                layer_output_sizes[i,:] = out_size                
                
            elif layers[i][0].upper() == 'L':
                layer, out_size = buildlinear(layers[i], layer_input_sizes[i,:].prod())
                self.model.append(layer)
                layer_output_sizes[i,:] = out_size

                if input_dimensions != 1:
                    input_dimensions = 1                    
                    routing[i] = ('v',) + routing[i]   #flag to apply View to input tensor

            elif layers[i][0].upper() == 'P':
                layer, out_size = buildpool(layers[i], layer_input_sizes[i,:])
                self.model.append(layer)
                layer_output_sizes[i,:] = out_size

            if layers[i][0].upper() == 'T':

                r = routing[i+1][1]
                out_sz = tuple(layer_output_sizes[i+r,1:])

                layer, out_size = buildtranspose(layers[i], layer_input_sizes[i,:], out_sz=out_sz)


                self.model.append(layer)                
                layer_output_sizes[i,:] = out_size

                
                
            elif layers[i][0].upper() == 'B':
                self.model.append(nn.BatchNorm2d(layer_input_sizes[i,0]))                
                layer_output_sizes[i,:] = layer_output_sizes[i-1, :]

            elif layers[i][0].upper() == 'R':
                #self.model.append(nn.ReLU())                
                self.model.append(nn.LeakyReLU(0.2))
                layer_output_sizes[i,:] = layer_output_sizes[i-1, :]

            elif layers[i][0].upper() == 'S':
                self.model.append(nn.Sigmoid())                     
                layer_output_sizes[i,:] = layer_output_sizes[i-1, :]     

            elif layers[i][0].upper() == 'D':
                self.model.append(nn.Dropout())                
                layer_output_sizes[i,:] = layer_output_sizes[i-1, :]
            elif layers[i][0].upper() == 'H':                
                self.model.append(nn.Tanh())
                layer_output_sizes[i,:] = layer_output_sizes[i-1, :]
            



        self.layer_input_sizes = layer_input_sizes
        self.layer_output_sizes = layer_output_sizes


    def forward(self, net_inputs):
        
        net_inputs = (net_inputs,)
        
        #array for storing output of every layer
        self.layer_outputs = []
        for i in range(len(self.model)):
            #iterate through layer

            layer_inputs = []            
            #retrieve inputs            
            for r in self.routing[i]:                
                if 'i' in str(r):                          
                    layer_inputs.append(net_inputs[int(r[1:])])
                elif str(r) == 'v':
                    continue
                elif r < 0:
                    layer_inputs.append(self.layer_outputs[i + r])
                    r = i + r
                else:
                    layer_inputs.append(self.layer_outputs[r])

                               
            

            #concatenate if more than 1 input
            if len(layer_inputs) > 1:   
                layer_input = torch.cat(layer_inputs, dim=1)                
            else:
                layer_input = layer_inputs[0]

            if self.routing[i][0] == 'v':
                layer_input = layer_input.view(layer_input.shape[0], -1)     


            self.layer_outputs.append(self.model[i](layer_input))
               



        return_outputs = []

        for output in self.model_outputs:
            o = self.layer_outputs[output]
            return_outputs.append(o)            

        return return_outputs

