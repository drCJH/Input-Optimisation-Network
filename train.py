#import libraries
import datetime
import torch
import torch.optim as optim
import torch.utils.data as tdata
import torch.nn as nn
from time import time
import math as maths
import random
import matplotlib.pyplot as plt
import network
import argparse


#import custom modules
from out import logger, checkpoint
import data
from net import Net
import cityscapes



#Arguments (name, default)
params = [
#dataset
("dsRoot", "../data/"),
("dspaths", "cityscapes/leftImg8bit/,cityscapes/leftImg8bit_rain/,cityscapes/leftImg8bit_foggyDBF/,A2D2/camera_lidar_semantic/"),
("sets", "train,val"),
("width", 384),
("height", 384),

#data augmentation
("flip", 0),
("mirror", 1),
("rotate", 0),
("crop", 0.1),
("noise", 0),
("balance", 1),

#general config
("runname", "ION_Deeplab_Train"),
("GPU", 0),
("saveoutput", 1),

#checkpoint loading
("cpfolder", ""),
("loadepoch", 0),
("segCP", "./checkpoints/deeplab_cityscapes"),

#Training parameters
("batchsize", 4),
("maxepochs", 1000),
("lr", 0.0001),
("lrdecay", 0),
("weightdecay", 0.000001),
("momentum", 0.9),
("jointopt", 0),

#net config
("ION", "unet"),
("targetnet", "deeplabv3")
]

#parse command line arguments
ap = argparse.ArgumentParser()
for p in params:    
    ap.add_argument("--" + p[0], type=type(p[1]), default=p[1])
args = ap.parse_args()


def run():
    torch.backends.cudnn.benchmark=True


    #create unique id for this run
    if args.runname and not args.runname[-1] == '_':
        args.runname = args.runname + '_' 
    runid = args.runname + datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    #create log file    
    o = logger("./output/" + runid, args.saveoutput)
    o.log("\n\n\nrun id: " + runid + '\n\n')

    #log arguments
    for arg in vars(args):
        o.log((arg, getattr(args, arg)))
    o.log("\n\n")


    #initalize datasets
    datasets = data.PrepDataSets(args)
    dataloaders = []  
    for d in datasets:
        dataloaders.append(tdata.DataLoader(d, batch_size=args.batchsize, shuffle=not d.isVal, num_workers=4, pin_memory=True))
    dsnames = args.sets.split(',')
    




    #initialise nets

    ION = Net(args.ION, o, input_size=datasets[0].input_size, output_size=[datasets[0].input_size[0]])
    net2 = Net(args.targetnet, o, input_size=datasets[0].input_size, output_size=[datasets[0].output_size[0]])

    lr = args.lr
    opt = optim.Adam(ION.parameters(), lr=lr, weight_decay=args.weightdecay)
    opt2 = optim.Adam(net2.parameters(), lr=lr, weight_decay=args.weightdecay)
    lossfn = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')




    


     



    #setup device
    if torch.cuda.device_count() == 0 or args.GPU < 0:  #no GPU or negative GPU argument passed        
        device = torch.device("cpu")        
        ION = ION.to(device)            
        net2 = net2.to(device)        
    elif torch.cuda.device_count() == 1:                #1 GPU is available, non negative GPU argument passed        
        device = torch.device("cuda:0")
        ION = ION.to(device)            
        net2 = net2.to(device)
    elif args.GPU < torch.cuda.device_count():          #>1 GPUs available, GPU index passed        
        device = torch.device("cuda:" + str(args.GPU))
        ION = ION.to(device)            
        net2 = net2.to(device)
    else:                                               #run in parallel across available GPUs                
        device = torch.device("cuda")        
        ION = nn.DataParallel(ION.to(device))                    
        net2 = nn.DataParallel(net2.to(device))

    o.log(("\nrunning on device:", device), include_time=True)


    #class for loading/saving checkpoints    
    cp = checkpoint(o, "./checkpoints/" + runid, args.loadepoch)

    if args.loadepoch != 0:  #load checkpoint        
        ION, opt = cp.load(ION, args.cpfolder, "ION", opt)
        if args.jointopt:
            net2, opt2 = cp.load(net2, args.cpfolder, "seg", opt2)

    if args.loadepoch == 0 or args.jointopt == 0:
        net2.model.load_state_dict(torch.load(args.segCP))
    


    #main loop
    startepoch = cp.epoch    
    lossvals = torch.zeros(len(datasets), args.maxepochs)
    net2.eval()

    while cp.epoch <= args.maxepochs:        

        if args.lrdecay:
            lr = args.lr * (args.lrdecay**cp.epoch)
            o.log("\nsetting learning rate to " + str(lr) + "\n")            
            
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            if args.jointopt:
                for param_group in opt2.param_groups:
                    param_group['lr'] = lr    
                  

        for d_i in range(len(dataloaders)):
            gradtype = torch.enable_grad()
            if datasets[d_i].isVal:                
                gradtype = torch.no_grad()            
                
            if datasets[d_i].isVal:
                ION.eval()
                net2.eval()
            else:
                ION.train()
                if args.jointopt:
                    net2.train()

            o.log(("\n-------------------------------"))
            o.log("run id: " + runid + '\n')
            o.log(("Epoch", cp.epoch), include_time=True)
            o.log(("Dataset:", dsnames[d_i]))                
            totiters = maths.ceil(len(datasets[d_i]) / args.batchsize)
            t0 = time()

            numiters = 0

            with gradtype:
                for b_i, batch in enumerate(dataloaders[d_i]):
                
                    opt.zero_grad()
                    opt2.zero_grad()

                    input, target, filenames = batch["image"], batch["target"], batch["filename"]
                    output = ION(input.to(device))                                               
                    segout = net2(output)
                    loss = lossfn(segout, target.to(device))                        
                    
                    lossvals[d_i, cp.epoch-1] += loss.item()

                    if not datasets[d_i].isVal:                            
                        loss.backward()
                        opt.step()
                        if args.jointopt:
                            opt2.step()
                        
                    #print stats every 1/10 epoch
                    if numiters > 0 and numiters % max(1, int(totiters / 10)) == 0:
                        print("Iteration " +  str(numiters) + " of " + str(totiters) + " Loss: " + '{0:.4f}'.format(loss.item())) #don't log, just print to console
                        

                        if args.saveoutput:
                            fname = 'E' + str(cp.epoch) + "_" + dsnames[d_i]                                
                            fname += "_i" + str(b_i)                                                            
                            
                            o.save_image(input[0,...], fname + "_input.png")
                            o.save_image(output[0,...], fname + "_output.png")                                
                            
                            outseg = cityscapes.displayseg(segout[0,:])
                            target = cityscapes.displayseg(target[0,:])
                            o.save_image(outseg, fname + "_outseg" + ".png")
                            o.save_image(target, fname + "_target" + ".png")

                    numiters += 1


            
            lossvals[d_i, cp.epoch-1] /= numiters
            
            
            o.log(("\nTotal iterations:", numiters), include_time=True)
            o.log(("Time taken:", o.formatTime(time() - t0)))                
            o.log(("loss:", '{0:.4f}'.format(lossvals[d_i, cp.epoch-1].item())))
            o.log("-------------------------------\n")

        #save checkpoint        
        cp.save(ION, opt, "ION")
        if args.jointopt:
            cp.save(net2, opt2, "seg")

        cp.epoch += 1

        #save loss curves
        if cp.epoch > 2:                                    
            o.plotloss(range(startepoch-1, cp.epoch-1), lossvals, dsnames)

    o.log(("finished after", args.maxepochs, "epochs"), include_time=True)
                    
        

if __name__ == '__main__':
    run()