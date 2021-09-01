1.# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 04:32:00 2021

@author: User
"""

import torch
import gc
import toyProblems
import train


'''
create dataset
'''
datasetTyp = "multiClasses" #"simple","multiClasses", "X-rays"
num_imgs = 6000

#info for "simple", "multiClasses" dataset
img_size = 64 # for yolo it has to be divisible by 64
min_object_size = 5
max_object_size = 20
num_objects = 1
randomNumObj = False # random number of ojects between 0 and num_objects
trainTestSplit = 0.8
num_shapes = 3 #not yet implemented for != 3
num_colors = 3 #not yet implemented for != 3


#creat/load data
train_X, test_X, train_y, test_y, test_imgs, test_bboxes, numChannels, test_shapes, test_colors = toyProblems.createDataset(num_imgs, 
                                    img_size, min_object_size, max_object_size, num_objects, trainTestSplit, datasetTyp, randomNumObj)
#plot 1 image
idxImagesToPlot = [0,1,2,3,4,5]
toyProblems.plotImgs(test_imgs, test_bboxes, test_shapes, test_colors, datasetTyp, idxImagesToPlot)




'''
train
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if not torch.cuda.is_available():
    print("cuda not available")
    
#hyperparameter
learning_rate = 0.0005
num_epochs = 5
batch_size = 16
dropout = 0.2
reproducibility = True
modelName = "simple" #simple, simpleCNN, yolo(not implemented yet)

model = train.train(train_X, test_X, train_y, test_y, device, numChannels, learning_rate, num_epochs, batch_size, dropout, 
                    reproducibility, modelName)



'''
show results
'''
# Predict bounding boxes on the test images.
pred_bboxes, pred_shapes, pred_colors = train.predict(test_X, model, device, num_objects, num_shapes, 
                                                       num_colors, batch_size, modelName, datasetTyp)


# plot results
idxImagesToPlot = [0,1,2]
toyProblems.plotImgs(test_imgs, test_bboxes, pred_shapes, pred_colors, datasetTyp, idxImagesToPlot,
                     pred_bboxes = pred_bboxes, pred = True)

#train.printMeanIOU(pred_bboxes, test_bboxes)


'''
clear cpu and gpu
'''
del train_X, test_X, train_y, test_y, test_imgs, test_bboxes, numChannels, test_shapes, test_colors
del model
del pred_bboxes, pred_shapes, pred_colors
torch.cuda.empty_cache()
gc.collect()

