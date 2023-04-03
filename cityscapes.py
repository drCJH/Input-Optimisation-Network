import numpy as np
from collections import namedtuple
import os

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'colour'])
classes = [
    CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence',                13, 255, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]


def idtotrainid(im0):
    #converts standard cityscapes label image to training label ids
    #im0 should be numpy array of dimensions (H,W)
    mapping = np.array([c.train_id for c in classes])
    return mapping[np.array(im0)]    


def displayseg(im0):
    #converts label image to colours for display
    #im0 should be either numpy array of (H,W) using training label IDs or pytorch tensor of (19,H,W)
    #returns numpy array of (H,W,3)

    if len(im0.shape) == 3:
        im1 = im0.cpu().detach().numpy().argmax(0)
        #im1 = im0.detach().numpy().argmax(0)   #use if pytorch tensor already on CPU
    else:
        im1 = im0
    im2 = np.zeros((im1.shape[0], im1.shape[1], 3), np.uint8)
    for i in range(len(classes)):
        t_id = classes[i][2]
        if t_id < 19:
            R = classes[i][7][0]
            G = classes[i][7][1]
            B = classes[i][7][2]
            im2[im1 == t_id, 0] = R
            im2[im1 == t_id, 1] = G
            im2[im1 == t_id, 2] = B
    return im2

def GetFilenames(rootdir, set):
    #retrieves paths for all images in the given dataset, assuming standard cityscapes directory structure
    #rootdir should be path to first level folder within cityscapes folder e.g. '.../cityscapes/leftImg8bit/'
    #set should be 'train', 'val', or 'test'

    trainIDs = False
    samples = []
    targets = []
    imgpath = rootdir + set + '/'
    lblpath = rootdir[:rootdir.find("/left")]+"/gtFine/" + set + '/'
    for folder in os.listdir(imgpath):
        for f in os.listdir(imgpath + folder):
            if f[-4:] == ".png":
                samples.append(imgpath + folder + '/' + f)               

                if len(targets) == 0:
                    #check if labels have already been reformatted with train IDs
                    f_lbl = f[:f.find("_left")] + "_gtFine_labelTrainIds.png"
                    if os.path.exists(lblpath + folder + '/' + f_lbl):
                        trainIDs = True

                if trainIDs:                    
                    f_lbl = f[:f.find("_left")] + "_gtFine_labelTrainIds.png"                    
                else:                    
                    f_lbl = f[:f.find("_left")] + "_gtFine_labelIds.png"
                
                targets.append(lblpath + folder + '/' + f_lbl)


    return (samples, targets, trainIDs)    


def metrics(results):
    #computes accuracy, precision, recall, intersection over union per class for tp/tn/fp/fn matrix
    #expected results: numpy array (19,4) containing tp, tn, fp, fn for each class (as output by eval())
    #returns numpy array (19,4) containing accuracy, precision, recall, intersection over union per class

    results2 = np.zeros((19, 4))
    for i in range(0, 19):
        
        tp = max(results[i][0], 0.01)
        tn = results[i][1]
        fp = results[i][2]
        fn = results[i][3]

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        IoU = tp / (tp + fp + fn)

        results2[i, 0] = accuracy
        results2[i, 1] = precision
        results2[i, 2] = recall
        results2[i, 3] = IoU

    return results2


def eval(target, output):
    #calculates true/false positives/negatives for each class in single output/target image pair
    #expected target: numpy array of dimensions(H,W) in long format
    #each element in target should be between 0 - 19 to denote class, ignored pixels should be 255
    #expected output: pytorch GPU tensor of dimensions (19,H,W) in float32 format
    #returns numpy array of dimesions (19,4) containing tp, tn, fp, fn for each class
    
    results = np.zeros((19, 4), dtype=np.float32)
    pred = output.argmax(0)

    n = (target != 255)
    for i in range(0, 19):        
        t = (target == i)
        o = (pred == i).cpu().detach().numpy()
        #o = (pred == i).detach().numpy()   #use if pytorch tensor already on CPU
        
        tp = (t & o) & n
        tn = ((t==0) & (o==0)) & n
        fp = ((t==0) & o) & n
        fn = (t & (o==0)) & n

        results[i, 0] = tp.sum()
        results[i, 1] = tn.sum()
        results[i, 2] = fp.sum()
        results[i, 3] = fn.sum()
    
    return results