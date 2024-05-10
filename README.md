# Deep-SM Instruction



## Introduction of Deep-SM
Deep-SM is used for improving the sensitivity of surface plasmon resonance microscopy (SPRM). Deep-SM is composed of a 
series of Residual UNet (ResUNet) and Bi-directional convolution long-short term n
image sequences.

Deep-SM can be applied to the analysis single particles with tiny size. We demonstrated Deep-SM on of gold nano-particle
 (5 nm and 10 nm) detection, extracellular particles counting, phase quantification and motion tracking in SPRM system. 
The results turned out that Deep-SM is a powerful tool for ultrasensitive nano-particle analysis.



## Deep-SM Implementation

### Data Preparation
The input data  should be cropped to a multiple of 16, as ResUNet down-samples 4 times. The number of frames in input
sequence should be equal or larger than four, which can be adjusted based on the quality of image sequences. The example 
data can be downloaded from the "data_example" folder, which includes the input SPRM sequence and corresponding GT
(average of 100 adjacent frames).

### Dependencies
Deep-SM depends on Pytorch, and should be implemented with Python 3.9 and above.

### Train Network
Run main.py after setting the environment. The models will be saved every 5 epochs in the "output" folder. The default 
parameters, such as learning rate = 1e-4, batch size = 8, can be adjusted according to the dataset.

### Test Network
Run DeepSM_Test.py if you would like to test the trained model on your data. We uploaded the trained Deep-SM model to 
Google Drive, and the shared link can be seen in 'Deep-SM Google Drive.txt'.