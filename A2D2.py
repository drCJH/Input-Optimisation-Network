import numpy as np
from collections import namedtuple
import os

A2D2Class = namedtuple('A2D2Class', ['colour', 'train_id'])
classes = [
    A2D2Class([255, 0, 0],26),
    A2D2Class([200, 0, 0],26),
    A2D2Class([150, 0, 0],26),
    A2D2Class([128, 0, 0],26),
    A2D2Class([182, 89, 6],33),
    A2D2Class([150, 50, 4],33),
    A2D2Class([90, 30, 1],33),
    A2D2Class([90, 30, 30],33),
    A2D2Class([204, 153, 255],24),
    A2D2Class([189, 73, 155],24),
    A2D2Class([239, 89, 191],24),
    A2D2Class([255, 128, 0],27),
    A2D2Class([200, 128, 0],27),
    A2D2Class([150, 128, 0],27),
    A2D2Class([0, 255, 0],32),
    A2D2Class([0, 200, 0],32),
    A2D2Class([0, 150, 0],32),
    A2D2Class([0, 128, 255],19),
    A2D2Class([30, 28, 158],19),
    A2D2Class([60, 28, 100],19),
    A2D2Class([0, 255, 255],20),
    A2D2Class([30, 220, 220],20),
    A2D2Class([60, 157, 199],20),
    A2D2Class([255, 255, 0],27),
    A2D2Class([255, 255, 200],27),
    A2D2Class([233, 100, 0],17),
    A2D2Class([110, 110, 0],7),
    A2D2Class([128, 128, 0],8),
    A2D2Class([255, 193, 37],7),
    A2D2Class([64, 0, 64],20),
    A2D2Class([185, 122, 87],12),
    A2D2Class([0, 0, 100],27),
    A2D2Class([139, 99, 108],8),
    A2D2Class([210, 50, 115],7),
    A2D2Class([255, 0, 128],0),
    A2D2Class([255, 246, 143],17),
    A2D2Class([150, 0, 150],7),
    A2D2Class([204, 255, 153],0),
    A2D2Class([238, 162, 173],13),
    A2D2Class([33, 44, 177],19),
    A2D2Class([180, 50, 180],7),
    A2D2Class([255, 70, 185],20),
    A2D2Class([238, 233, 191],6),
    A2D2Class([147, 253, 194],21),
    A2D2Class([150, 150, 200],9),
    A2D2Class([180, 150, 200],8),
    A2D2Class([72, 209, 204],1),
    A2D2Class([200, 125, 210],7),
    A2D2Class([159, 121, 238],20),
    A2D2Class([128, 0, 255],7),
    A2D2Class([255, 0, 255],7),
    A2D2Class([135, 206, 255],23),
    A2D2Class([241, 230, 255],11),
    A2D2Class([96, 69, 143],0),
    A2D2Class([53, 46, 82],0),
]

def colourtotrainid(im0):
    im1 = np.full((im0.shape[0], im0.shape[1]), 255, dtype=np.uint8)    
    for c in classes:
        col = c.train_id
        if col > 18:
            col = 255
        im1[(im0[:,:,0] == c.colour[0]) & (im0[:,:,1] == c.colour[1]) & (im0[:,:,2] == c.colour[2])] = col
    return im1

def GetFilenames(rootdir, set):    
    samples = []
    targets = []
    imgpath = rootdir + set + '/'
    #lblpath = rootdir[:rootdir.find("/cam")] + "/labels/" + set + '/'
    for folder in os.listdir(imgpath):
        folder += "/camera/cam_front_center"
        for f in os.listdir(imgpath + folder):
            if f[-4:] == ".png":
                samples.append(imgpath + folder + '/' + f)                                                              
                f_lbl = f.replace("_camera_fr", "_label_fr")
                #targets.append(lblpath + f_lbl)
                folder_lbl = folder.replace("/camera/", "/label/")
                targets.append(imgpath + folder_lbl + '/' + f_lbl)
    return (samples, targets)    
