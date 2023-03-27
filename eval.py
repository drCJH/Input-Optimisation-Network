import subprocess
import os

#ds = ["cs_", "f_", "r_", "a_", "d_"]
#datasets = ["cityscapes/leftImg8bit/val", "cityscapes/leftImg8bit_foggyDBF/val", "cityscapes/leftImg8bit_rain/val", "A2D2/images/val", "cityscapes/leftImg8bit_dark/val", ]

ds = ["f_", "r_", "a_", "d_"]
datasets = ["cityscapes/leftImg8bit_foggyDBF/val", "cityscapes/leftImg8bit_rain/val", "A2D2/images/val", "cityscapes/leftImg8bit_dark/val", ]



rn = "ION_JO_1_"
IONcpf = "./checkpoints/ION_JO_1_220107_183406/"
segcpf = "./checkpoints/ION_JO_1_220107_183406/"
arch = "unet"


rf = open("results_" + rn, 'w')
rf.close()

for i in range(len(ds)):
    
    e = 300
    print("Checking for " + IONcpf + "epoch_" + str(e) + "_G")
    while os.path.exists(IONcpf + "epoch_" + str(e) + "_G"):
        args1 = ""
        args1 += " --loadepoch=" + str(e)
        args1 += " --runname=" + ds[i] + rn + str(e)
        args1 += " --cpfolder=" + IONcpf
        args1 += " --archs=" + arch
        args1 += " --datasets=/home/cjh9975/data/" + datasets[i]
        args1 += " --resultsfile=" + "results_" + rn

        s1 = "python3 ./autotest.py" + args1
        print("\n\n" + s1 + "\n\n")
        subprocess.call(s1, shell=True)



        rf = open("results_" + rn, 'r')
        runid = rf.readlines()[-1].split('\t')[0]
        rf.close()

        if ds[i] == 'a_':
            labdir = "/home/cjh9975/data/A2D2/labels/val/"
        else:
            labdir = "/home/cjh9975/data/cityscapes/gtFine/val/"


        args2 = ""
        args2 += " --runname=seg_" + ds[i] + rn + str(e)
        args2 += " --imdir=../unetDLseg2/output/test/" + runid
        args2 += " --labdir=" + labdir
        args2 += " --checkpoint=../unetDLseg2/" + segcpf + "epoch_" + str(e) + "_2"        
        args2 += " --resultsfile=" + "../unetDLseg2/results_" + rn

        s2 = "python3 ../segmentation/test0.py" + args2       

        print("\n\n" + s2 + "\n\n")
        subprocess.call(s2, shell=True)
        
        e += 100

