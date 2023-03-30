#import libraries
import os
from PIL import Image, ImageOps
import random
import torch
import torch.utils.data as data
import numpy as np
import math as maths
import matplotlib.pyplot as plt
import matplotlib.colors as colours

#import custom modules
from out import logger
import cityscapes
import A2D2


def display(images):

    fig=plt.figure(figsize=(8, 8))

    columns = len(images)
    rows = 1
    for i in range(len(images)):
        im = images[i].numpy()
        if im.shape[0] == 3:
            im = im.transpose(1,2,0)            

        fig.add_subplot(rows, columns, i+1)
        plt.imshow(im)
    plt.show()    


def preprocess(image, args, target, isVal=False):
    
    if not isVal:   #only augment training set        
        if args.mirror:
            if random.randint(0,1) == 1:
                image = ImageOps.mirror(image)                
                target = ImageOps.mirror(target)

        if args.flip:        
            if random.randint(0,1) == 1:
                image = ImageOps.flip(image)  
                target = ImageOps.flip(target)   
                
        if args.rotate:
            if random.randint(0,1) == 1:    #rotate images 50% of the time            
                r = (random.random() - 0.5) * args.rotate * 2   #random amount between -args.rotate and +args.rotate
                image = image.rotate(r, expand=1, resample=Image.BICUBIC)
                target = target.rotate(r, expand=1, resample=Image.NEAREST)                

        if args.crop:
            x0 = random.randint(0, int(image.size[0] * args.crop))
            y0 = random.randint(0, int(image.size[1] * args.crop))        
            x1 = image.size[0] - random.randint(0, int(image.size[0] * args.crop))
            y1 = image.size[1] - random.randint(0, int(image.size[1] * args.crop))
            image = image.crop((x0, y0, x1, y1))            
            target = target.crop((x0, y0, x1, y1))
            
    if args.width > 0 and args.height > 0:
        #random crop of correct aspect ratio
        w, h = w2, h2 = cx1, cy1 = image.size
        cx0 = cy0 = 0    

        if w/args.width > h/args.height:
            w2 = h * (args.width / args.height)
            cx0 = random.randint(0, w-w2)
            cx1 = cx0 + w2
        elif h/args.height > w/args.width:
            h2 = w * (args.height / args.width)
            cy0 = random.randint(0, h-h2)
            cy1 = cy0 + h2

        image = image.crop((cx0, cy0, cx1, cy1))        
        target = target.crop((cx0, cy0, cx1, cy1))
        
    width2 = height2 = 0
    #resize
    if args.width > 0 and args.height > 0:
        #both dimensions given
        width2 = args.width
        height2 = args.height
    elif args.height > 0:
        #maintain aspect ratio based on new height
        height2 = args.height
        width2 = image.size[0] * (height2 / image.size[1])
    elif args.width > 0:
        #maintain aspect ratio based on new width
        width2 = args.width
        height2 = image.size[1] * (width2 / image.size[0])

    if width2 * height2 > 0:
        image = image.resize((int(width2), int(height2)), resample=Image.BICUBIC)       
        target = target.resize((int(width2), int(height2)), resample=Image.NEAREST)
            
    #convert to np array
    image = np.array(image).astype(np.float32) / 255
    target = np.array(target) 
    
    if not isVal and args.noise:            
        n = np.random.normal(loc=1, scale=random.uniform(0,1) * args.noise, size=image.shape).astype(np.float32)
        image = image * n
    
    return image, target
   

def PrepDataSets(args):
    datasets = []
    sets = args.sets.split(',')
    for s in sets:
        datasets.append(segmentation_dataset(args, s))
    return datasets


class segmentation_dataset(data.Dataset):

    def __init__(self, args, set):

        dsroot = args.dsRoot
        dspaths = args.dspaths.split(',')

        self.isVal = set in ["val", "test"]        
        self.args = args
        self.samples = []
        self.targets = []
        self.trainIDs = []

        for i in range(len(dspaths)):
            imgs, lbls, tIDs = [], [], False
            if "A2D2" in dspaths[i]:
                imgs, lbls = A2D2.GetFilenames(dsroot + dspaths[i], set)
            elif "cityscapes" in dspaths[i]:
                imgs, lbls, tIDs = cityscapes.GetFilenames(dsroot + dspaths[i], set)
            self.samples.append(imgs)
            self.targets.append(lbls)
            self.trainIDs.append(tIDs)
             
        #retrieve sample to check expected input/output size 
        self.input_size = self.__getitem__(0)["image"].shape
        self.output_size = self.__getitem__(0)["target"].shape


    def __len__(self):
        #return length of largest subset
        maxlen = 0
        for s in self.samples:
            maxlen = max(maxlen, len(s))
        return maxlen
            

    def __getitem__(self, idx):
        #select subset at random so each gets balanced use regardless of size
        ss = random.randint(0, len(self.samples) - 1)
        while idx >= len(self.samples[ss]):
            idx = random.randint(0, len(self.samples[ss]) - 1)
        
        image = Image.open(self.samples[ss][idx])               
        target = Image.open(self.targets[ss][idx])   

        image, target = preprocess(image, self.args, target, isVal=self.isVal)                
        image = torch.from_numpy(image.transpose(2, 0, 1))

        if not self.trainIDs[ss]:
            if ("cityscapes" in self.targets[ss][idx]):
                target = cityscapes.idtotrainid(target)
            elif ("A2D2" in self.targets[ss][idx]):
                target = A2D2.colourtotrainid(target)

        target = torch.from_numpy(target).long()

        #display([image, target])
        return {"image": image, "target": target, "filename": self.samples[ss][idx]}