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
("dspaths", "cityscapes/leftImg8bit/"),
#("dspaths", "cityscapes/leftImg8bit_rain/"),
#("dspaths", "cityscapes/leftImg8bit_foggyDBF/"),
#("dspaths", "A2D2/camera_lidar_semantic/"),
("sets", "val"),
("width", 768),
("height", -1),


#general config
("runname", "ION_Deeplab_Test"),
("GPU", 0),
("saveoutput", 1),

#checkpoint loading
("cpfolder", "ION_Deeplab_Train_230330_134815"),
("loadepoch", 2),
("segCP", "./deeplab_cityscapes"),

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

    #load checkpoint        
    ION, opt = cp.load(ION, args.cpfolder, "ION", opt)
    if args.jointopt:
        net2, opt2 = cp.load(net2, args.cpfolder, "seg", opt2)

    if args.loadepoch == 0 or args.jointopt == 0:
        net2.model.load_state_dict(torch.load(args.segCP))
    


    #main loop
    ION.eval()
    net2.eval()


    for d_i in range(len(dataloaders)):    
        gradtype = torch.no_grad()            
            

        o.log(("\n-------------------------------"))
        o.log("run id: " + runid + '\n')
        o.log(("Dataset:", dsnames[d_i]))                
        totiters = maths.ceil(len(datasets[d_i]) / args.batchsize)
        t0 = time()

        numiters = 0

        with gradtype:
            for b_i, batch in enumerate(dataloaders[d_i]):            

                input, target, filenames = batch["image"], batch["target"], batch["filename"]
                output = ION(input.to(device))                                               
                segout = net2(output)
                    
                #print stats every 1/10 epoch
                if numiters > 0 and numiters % max(1, int(totiters / 10)) == 0:
                    print("Iteration " +  str(numiters) + " of " + str(totiters)) #don't log, just print to console
                    

                if args.saveoutput:
                    for i in range(len(filenames)):
                        o.save_image(input[i,...], filenames[i] + "_input.png")
                        o.save_image(output[i,...], filenames[i] + "_output.png")                                
                        
                        outseg = cityscapes.displayseg(segout[i,:])
                        target = cityscapes.displayseg(target[i,:])
                        o.save_image(outseg, filenames[i] + "_outseg" + ".png")
                        o.save_image(target, filenames[i] + "_target" + ".png")

                numiters += 1

        
        o.log(("\nTotal iterations:", numiters), include_time=True)
        o.log(("Time taken:", o.formatTime(time() - t0)))
        o.log("-------------------------------\n")

       

    o.log(("finished testing"), include_time=True)
                    
        

if __name__ == '__main__':
    run()