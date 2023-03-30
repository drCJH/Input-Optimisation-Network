#import libraries
import datetime
import os
import torch
from matplotlib import pyplot as plt
import imageio
import numpy as np



class logger():

    def __init__(self, logloc, saveoutput=False):
        #create directory if it doesn't exist
        self.outfolder = logloc[:logloc.rfind('/')]
        if not os.path.exists(self.outfolder):
            os.makedirs(self.outfolder)

        self.logloc = logloc
        self.logpath = logloc + "_log.txt"

        if saveoutput:
            self.imfolder = logloc + "/"
            if not os.path.exists(self.imfolder):
                os.makedirs(self.imfolder)
    
    def formatTime(self, t):        
        hours = int(t/3600)
        minutes = int((t%3600)/60)
        seconds = t%60

        return str(hours) + ':' + str(minutes) + ':' + '{0:.3f}'.format(seconds)

    def log(self, text, toConsole=True, include_time=False):
        #check if text is list/tuple of items or single item
        if type(text) in [type(()), type([])]:
            s = ""
            for i in text:
                s = s + str(i) + " "
        else:            
            s = str(text)

        if toConsole:
            print(s)

        #file is opened and closed for each line so it can be read immediately
        f = open(self.logpath, 'a')
        if include_time:    #log time at start of line
            while s[0] == '\n': #if line starts with newline(s) write before time
                f.write('\n')
                s = s[1:]
            f.write(datetime.datetime.now().strftime("%y%m%d_%H%M%S:  "))
        f.write(s + '\n')
        f.close()





    def save_image(self, im, filename):

        if len(im.shape) == 4: #whole batch passed, just take first sample
            im = im[0,...]

        if type(im) == type(torch.tensor(0)):
            #to cpu/numpy
            im = im.cpu().detach().numpy()
            #C dimension
            if len(im.shape) == 3:
                if im.shape[0] == 3:
                    im = im.transpose(1, 2, 0)
                else:
                    im = im[0,...]
        

        if im.dtype == np.float32:
            #normalise
            im -= im.min()
            im /= im.max()
            
            imageio.imwrite(self.imfolder + filename, (im * 255).astype(np.uint8))
        else:
            imageio.imwrite(self.imfolder + filename, im.astype(np.uint8))



        



    def plotloss(self, eprange, loss, datanames):
        colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        ax = []
        colidx = 0
        for i in range(loss.shape[0]):        
            lbl = datanames[i]                                
                
            ax.append(plt.plot(range(eprange[0]+1, eprange[-1]+2), loss[i,eprange], color=colours[colidx], label=lbl))
            colidx += 1

        plt.legend()
        plt.ylabel('loss')
        plt.yscale('log')        
        plt.xlabel("epochs")
        plt.savefig(self.logloc + "_epoch_" + str(eprange[-1]+1) + "_loss.png")
        plt.cla()

        if len(eprange) > 1 and os.path.exists(self.logloc + "_epoch_" + str(eprange[-1]) + "_loss.png"):
            os.remove(self.logloc + "_epoch_" + str(eprange[-1]) + "_loss.png") #remove previous epoch plot



class checkpoint():

    def __init__(self, log, cpfolder = "/", epoch = 0, keepevery=10, gpu=0):
         #epoch to start from
        if epoch == 0:
            self.epoch = 1
        else:
            self.epoch = epoch

        #directory to save checkpoints
        self.dir = cpfolder
        if self.dir[-1] != '/':
            self.dir += '/'
        #create directory
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
       
        self.keep = keepevery #frequency of checkpoints to keep
        self.log = log

        self.gpu = gpu



    def save(self, net, opt=None, identifier=""):        
            #save network weights
            torch.save(net.model.state_dict(), self.dir + "epoch_" + str(self.epoch) + identifier)
            #save optimiser if passed
            if opt != None:
                torch.save(opt.state_dict(), self.dir + "optim_" + str(self.epoch) + identifier)

            #delete previous checkpoint
            if self.epoch % self.keep != 1:
                if os.path.exists(self.dir + "epoch_" + str(self.epoch-1) + identifier):
                    os.remove(self.dir + "epoch_" + str(self.epoch-1) + identifier)
                if os.path.exists(self.dir + "optim_" + str(self.epoch-1) + identifier):
                        os.remove(self.dir + "optim_" + str(self.epoch-1) + identifier)

             



    def load(self, net, cpfolder, identifier="", opt=None):

        #load checkpoint
        
        self.log.log("loading checkpoint:" + cpfolder + "epoch_" + str(self.epoch) + identifier)
        net.model.load_state_dict(torch.load(cpfolder + "epoch_" + str(self.epoch) + identifier))        
        
        if opt != None:            
            self.log.log("loading optimizer:" + cpfolder + "optim_" + str(self.epoch) + identifier)
            opt.load_state_dict(torch.load(cpfolder + "optim_" + str(self.epoch) + identifier))            
                                
        self.epoch += 1        

        return net, opt
            


        


