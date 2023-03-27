#import libraries
import datetime
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from time import time
import math as maths
import random
import matplotlib.pyplot as plt


#import custom modules
from out import logger, checkpoint
from data import *
import arguments
from net import Net


#aguments (gname, default, min, max)
params = [
#(name, default, min, max, help)
#or (name, default, help)

#dataset parameters
("datasets", "/home/cjh9975/data/cityscapes/leftImg8bit/val"),
("makedark", 0),
#("datasets", "/home/cjh9975/data/A2D2/images/val"),


("dsnames", "test"),

#("datasets", "C:/datasets/exposure/data/small/"),
#("dsnames", "small"),

("width", 768),
("height", -1),

#general config
("runname", "cs_ION_ft_300"),
("config", "./config.txt"),
("GPU", 1),
("saveoutput", "1"),
("outdir", "./output/"),

#checkpoint loading
#("cpfolder", "D:/experiments/ION/checkpoints/all_ft_all_201121_165612/"),
#("loadepoch", 200),

("cpfolder", "../checkpoints/ION_ft/"),
("loadepoch", 300),





#net config
("archs", "unet"),
("identifiers", "G"),
("batchsize", 1, 1, None),

]





def run():    #avoids multithreading problems on windows
    #parse aguments
    args = arguments.parse_args(params)

    #create unique id for this run
    if args.runname and not args.runname[-1] == '_':
        args.runname = args.runname + '_' 
    runid = args.runname + datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    #create log file    
    o = logger(args.outdir + runid, args.saveoutput)
    o.log("\n\n\nrun id: " + runid + '\n\n')

    for arg in vars(args):
        o.log((arg, getattr(args, arg)))
    o.log("\n\n")


    #initalize datasets
    datasets = []
    dataloaders = []  

    datapaths = args.datasets.split(',')
    dsnames = args.dsnames.split(',')

    
    for i in range(len(datapaths)):
        datapaths[i] = datapaths[i].strip()
        dsnames[i] = dsnames[i].strip()
        #datasets.append(classification_dataset(datapaths[i], args, o))
        #datasets.append(regression_dataset(datapaths[i], args, o))
        #datasets.append(segmentation_dataset(datapaths[i], args, o))
        #datasets.append(imagetoimage_dataset(datapaths[i], args, o))
        
        #datasets.append(unsupervised_dataset(datapaths[i], args, o, istest=True))
        #datasets.append(ready_made_unsupervised_dataset(datapaths[i], datapaths[i].replace("images", "labels"), args, o, istest=True))
        
        

        #datasets.append(ready_made_unsupervised_dataset(datapaths[i], "", args, o, istest=True))
        
        
        #datasets.append(ready_made_unsupervised_dataset(datapaths[i].replace("images", "foggy"), "", args, o, istest=True))
        datasets.append(ready_made_unsupervised_dataset(datapaths[i], "", args, o, istest=True, makedark=args.makedark))
        #datasets.append(ready_made_unsupervised_dataset(datapaths[i].replace("cityscapes", "audi"), "", args, o, istest=True))
        
        
        o.log(("dataset '" + dsnames[i] + "' has", len(datasets[-1]), "samples"), include_time=True)    
        dataloaders.append(data.DataLoader(datasets[-1], batch_size=args.batchsize, shuffle=False, num_workers=8))
    



    #initialise net(s)
    nets = []   #lists in case more than one required eg GAN
    identifiers = []
    
    archs = args.archs.lower().split(',')
    identifiers = args.identifiers.split(',')
    
    for i in range(len(archs)):    #assume same number of nets, opts and losses

        archs[i] = archs[i].strip()               
        identifiers[i] = identifiers[i].strip()        

        #nets.append(Net(archs[i], o, input_sizes=datasets[0].input_sizes, output_sizes=datasets[0].output_sizes))
        
        #different output sizes for generator and discriminator
        
        #nets.append(Net(archs[i], o, input_sizes=[(3, 384, 384)], output_sizes=[datasets[0].output_sizes[0]]))
        nets.append(Net(archs[i], o, input_sizes=[datasets[0].input_sizes[0]], output_sizes=[datasets[0].output_sizes[0]]))
        
        

    if len(identifiers) < len(nets):
        for i in range(len(identifiers), len(nets)):
            identifiers.append("_" + str(i))    




    #setup device
    if torch.cuda.device_count() == 0 or args.GPU < 0:  #no GPU or negative GPU argument passed        
        device = torch.device("cpu")
    elif torch.cuda.device_count() == 1:                #1 GPU is available, non negative GPU argument passed        
        device = torch.device("cuda:0")
    elif args.GPU < torch.cuda.device_count():          #>1 GPUs available, GPU index passed        
        device = torch.device("cuda:" + str(args.GPU))
    else:                                               #run in parallel across available GPUs                
        device = torch.device("cuda")
        net = nn.DataParallel(net.to(device))
    o.log(("\nrunning on device:", device), include_time=True)

    for n in nets:
        n.to(device)


    #class for loading/saving checkpoints    
    cp = checkpoint(o, epoch=args.loadepoch)
    
    
    #nets = cp.load(nets, args.cpfolder, identifiers[0]) #load checkpoint
    #if args.cpfolder and args.cpdir not in args.cpfolder:
        #args.cpfolder = args.cpdir + args.cpfolder
    if args.loadepoch != 0:
        for i in range(len(nets)):
            nets[i] = cp.load(nets[i], args.cpfolder, identifiers[i]) #load checkpoint


    for d_i in range(len(dataloaders)):

        for n_i in range(len(nets)):                
            nets[n_i].eval()            

            o.log(("\n-------------------------------"))
            o.log("run id: " + runid + '\n')            
            o.log(("Dataset:", dsnames[d_i]))
            o.log(("Net:", archs[n_i], identifiers[n_i], "\n"))
            totiters = maths.ceil(len(datasets[d_i]) / args.batchsize)
            t0 = time()

            numiters = 0

            with torch.no_grad():
                for b_i, batch in enumerate(dataloaders[d_i]):
                    inputs, targets, filenames = batch["image"], batch["target"], batch["filename"]

                    
                    outputs = nets[n_i]([inp.to(device) for inp in inputs])

                    #metrics = eval_metrics(outputs, targets)    #in case evaluation metrics are required

                    if b_i > 0 and b_i % max(1, int(totiters / 20)) == 0:
                        print("Iteration " +  str(b_i) + " of " + str(totiters)) #don't log, just print to console
                        
                    fname = filenames[0][filenames[0].rfind('/')+1:]
                    dataset=True
                    if dataset:
                        o.save_image(outputs[i][0,...], fname)
                    else:
                        o.save_image(inputs[i][0,...],  fname + "_input"  + ".png")
                        o.save_image(outputs[i][0,...], fname + "_output" + ".png")
                        #o.save_image(targets[i][0,...], fname + "_target" + ".png")
                        

            
            #o.log(("\nTotal iterations:", totiters), include_time=True)
            o.log(("\nTotal iterations:", numiters), include_time=True)
            o.log(("Time taken:", o.formatTime(time() - t0)))                
            o.log("-------------------------------\n")


    o.log(("finished testing"), include_time=True)
                    
        
#avoids multithreading problems on windows    
if __name__ == '__main__':
    run()