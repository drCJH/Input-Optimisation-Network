#import libraries
import datetime
import torch
import torch.utils.data as tdata
import torch.nn as nn
from time import time
import math as maths
import argparse
import numpy as np


#import custom modules
from out import logger
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
("height", 384),

#general config
("runname", "ION_Deeplab_Test"),
("GPU", 0),
("saveoutput", 1),
("batchsize", 4),

#checkpoint loading
("cpfolder", "ION_Deeplab_Train_230330_134815/"),
("loadepoch", 2),
("segCP", "./checkpoints/deeplab_cityscapes"),
("jointopt", False),

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


   

    ION.model.load_state_dict(torch.load("./checkpoints/" + args.cpfolder + "epoch_" + str(args.loadepoch) + "ION"))
    if args.jointopt:
        net2.model.load_state_dict(torch.load("./checkpoints/" + args.cpfolder + "epoch_" + str(args.loadepoch) + "seg"))
    else:
        net2.model.load_state_dict(torch.load(args.segCP))
    


    ION.eval()
    net2.eval()
    
    #total metrics across dataset
    results = np.zeros((19, 8), dtype=np.float32)
    #metrics of each image, list of tuples
    immetrics = []
    
    #main loop
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

                for i in range(input.shape[0]):
                    imstats = cityscapes.eval(target[i], segout[i])
                    imstats2 = cityscapes.metrics(imstats)
                    immetrics.append((filenames[i][filenames[i].rfind('/')+1:filenames[i].rfind('.')], imstats2[:,0].mean(), imstats2[:,1].mean(), imstats2[:,2].mean(), imstats2[:,3].mean()))                
                    results[:,:4] += imstats
                    
                #print progress every 1/10 epoch
                if numiters > 0 and numiters % max(1, int(totiters / 10)) == 0:
                    print("Iteration " +  str(numiters) + " of " + str(totiters)) #don't log, just print to console
                    
                if args.saveoutput:
                    for i in range(len(filenames)):
                        fn = filenames[i][filenames[i].rfind('/')+1:filenames[i].rfind('.')]
                        o.save_image(input[i,...], fn + "_input.png")
                        o.save_image(output[i,...], fn + "_output.png")                                
                        
                        out_seg = cityscapes.displayseg(segout[i,...])
                        out_target = cityscapes.displayseg(target[i,...])
                        o.save_image(out_seg, fn + "_outseg" + ".png")
                        o.save_image(out_target, fn + "_target" + ".png")

                numiters += 1

        
        o.log(("\nTotal iterations:", numiters), include_time=True)
        o.log(("Time taken:", o.formatTime(time() - t0)))
        o.log("-------------------------------\n")


    #log metrics for each class across dataset
    results[:,4:] = cityscapes.metrics(results)
    f = open("./output/" + runid + "/" + "results.txt", 'w')
    f.writelines(["class,tp,tn,fp,fn,accuracy,precision,recall,IoU\n"])

    for ii in range(1, len(cityscapes.classes)):
        if cityscapes.classes[ii][2] < 19:
            i = cityscapes.classes[ii][2]
            line = cityscapes.classes[ii][0] #class name
            for j in range(8):
                numstr = '{0:.4f}'.format(results[i][j])
                while (numstr[-1] == '0' or numstr[-1] == '.') and numstr.find('.') != -1:
                    numstr = numstr[:-1]
                line = line + "," + numstr
            line = line + "\n"

            f.writelines([line])

    #compute mean/total across all classes
    line = "mean"
    for j in range(8):        
        tot = 0.0
        tot = results[:,j].sum()
        if j > 3:
            tot /= (19)           
        numstr = '{0:.4f}'.format(tot)       
        while (numstr[-1] == '0' or numstr[-1] == '.') and numstr.find('.') != -1:
            numstr = numstr[:-1]        
        line = line + "," + numstr

    f.writelines([line])
    f.close()
    print("Results: " + line)


    #log metrics for individual images
    f = open("./output/" + runid + "/" + "results_by_image.txt", 'w')
    f.writelines(["filename,accuracy,precision,recall,IoU\n"])
 
    for i in range(0, len(immetrics)):    
        line = immetrics[i][0]

        for j in range(1, 5):
            numstr = '{0:.6f}'.format(immetrics[i][j])
            while (numstr[-1] == '0' or numstr[-1] == '.') and numstr.find('.') != -1:
                numstr = numstr[:-1]
            line = line + "," + numstr
        line = line + "\n"

        f.writelines([line])
    f.close()

    o.log(("finished testing"), include_time=True)
                    
        

if __name__ == '__main__':
    run()