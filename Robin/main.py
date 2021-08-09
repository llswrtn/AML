# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 04:32:00 2021

@author: User
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import sys 
import os
import matplotlib.pyplot as plt
import matplotlib

import toyProblems
import nets
import train


print("setup train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if not torch.cuda.is_available():
    print("cuda not available")
        



###
# create dataset
num_imgs = 5000
img_size = 448
min_object_size = 4
max_object_size = 16
num_objects = 3
trainTestSplit = 0.8


train_X, test_X, train_y, test_y, test_imgs, test_bboxes, numChannels, test_shapes, test_colors = toyProblems.createSimpleDataSet(num_imgs, 
                                    img_size, min_object_size, max_object_size, num_objects, trainTestSplit)
toyProblems.plotImgSimpleDataset(test_imgs[1], test_bboxes[1])


#train_X, test_X, train_y, test_y, test_imgs, test_bboxes, numChannels, test_shapes, test_colors = toyProblems.createMedicoreDataSet(num_imgs, 
#                                    img_size, min_object_size, max_object_size, num_objects, trainTestSplit)
#toyProblems.plotImgMedicoreDataset(test_imgs[1], test_bboxes[1], test_shapes[1], test_colors[1])




###
# train
learning_rate = 0.05
num_epochs = 30
batch_size = 256
dropout = 0.2
reproducibility = True
modelName = "simpleCNN" #simple, simpleCNN, yolo(not implemented yet)

model = train.train(train_X, test_X, train_y, test_y, numChannels, learning_rate, num_epochs, batch_size, dropout, 
                    reproducibility, modelName)


###
# Predict bounding boxes on the test images.
pred_bboxes = train.predict(test_X, model, device, num_objects)



###
# plot results
def plotSomeResults():
    plt.figure(figsize=(12, 3))
    for i_subplot in range(1, 5):
        plt.subplot(1, 4, i_subplot)
        i = np.random.randint(len(test_imgs))
        plt.imshow(test_imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
        for pred_bbox, exp_bbox in zip(pred_bboxes[i], test_bboxes[i]):
            plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r', fc='none'))
            plt.annotate('IOU: {:.2f}'.format(train.IOU(pred_bbox, exp_bbox)), (pred_bbox[0], pred_bbox[1]+pred_bbox[3]+0.2), color='r')
            
def printMeanIOU():
    # Calculate the mean IOU (overlap) between the predicted and expected bounding boxes on the test dataset. 
    summed_IOU = 0.
    for pred_bbox, test_bbox in zip(pred_bboxes.reshape(-1, 4), test_bboxes.reshape(-1, 4)):
        summed_IOU += train.IOU(pred_bbox, test_bbox)
    mean_IOU = summed_IOU / len(pred_bboxes)
    print("mean IOU: ", mean_IOU)
'''  
def plotMediResults():
    plt.figure(figsize=(16, 8))
    for i_subplot in range(1, 9):
        plt.subplot(2, 4, i_subplot)
        i = np.random.randint(len(test_X))
        plt.imshow(test_imgs[i], interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
        for bbox, shape, color in zip(pred_bboxes[i], pred_shapes[i], pred_colors[i]):
            plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='k', fc='none'))
            plt.annotate(shape_labels[shape], (bbox[0], bbox[1] + bbox[3] + 0.7), color=color_labels[color], clip_on=False, bbox={'fc': 'w', 'ec': 'none', 'pad': 1, 'alpha': 0.6})
'''
plotSomeResults()
printMeanIOU()
