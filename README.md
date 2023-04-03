# Input-Optimisation-Network
Learned image preprocessing for semantic segmentation, published in the following 2 papers:

Holder, Christopher J., Majid Khonji, and Jorge Dias. "Input optimisation network for semantic segmentation of underexposed images." IEEE International Symposium on Safety, Security, and Rescue Robotics. 2020. (https://ieeexplore.ieee.org/abstract/document/9292626)

Holder, Christopher J., Majid Khonji, Jorge Dias, and Mohammed Shafique. "Building Resilience to Out-of-Distribution Visual Data via Input Optimization and Model Finetuning." arXiv preprint arXiv:2211.16228. 2022  (https://arxiv.org/abs/2211.16228)

The goal of the Input Optimisation Network (ION) is to learn an image preprocessing function optimised for a specific target model. During training (illustrated in the diagram below) image *x* is input to ION *G* which outputs modified image *x'* which is subsequently input to target model *F*. The output of *F*, in this case segmentation class probability map *y'*, is used to compute loss via function *L*, which is then used to update the weights of *G* while those of *F* remain fixed.

![flow](https://user-images.githubusercontent.com/93485988/229435117-766c2a05-85f7-4987-ad72-fb7c1f6d4b3d.png)

The aim is that the ION learns to generate outputs that are optimised to achieve the best performance of the target model.

## Models
This implementation enables training of a u-net based ION with the target model a Deeplab v3 with mobilenet backbone trained for semantic segmentation with the Cityscapes dataset. The checkpoint for the segmentation model can be download from https://drive.google.com/file/d/1SYRPZZQ4_IJb6xxDCGigt2weoMHb3yWt/ and should be placed in the checkpoints folder.
More compact u-net models can be used by setting the --ION argument to "unets" (smaller) and "unetss" (smallest).

## Datasets
The ION can be trained to preprocess the Cityscapes dataset as well as the cs_rain and cs_fog version that have been modified with simulated weather effects, and the Audi Autonomous Driving Dataset (A2D2) Semantic Segmentation dataset. the target segmentation model has only been trained on the original Cityscapes data, so cs_fog, cs_rain and A2D2 can all be considered out of distribution.

Cityscapes, cs_fog and cs_rain can be download from https://www.cityscapes-dataset.com/

A2D2 can be downloaded from https://www.a2d2.audi/a2d2/en/download.html

The function colourtotrainid in A2D2.py is used to align A2D2 labels with the Cityscapes label schema.

## Training
run train.py to train the ION, using the --dsRoot and --dspaths arguments to point to the appropriate dataset paths.

## Testing
Once the ION is trained, test.py can be used for evaluation. Only 1 dataset can be evaluated at a time, so use the --dspaths argument to set the appropriate dataset. Set --useION=0 to set a baseline by passing the input image directly to the segmentation model without preprocessing.
