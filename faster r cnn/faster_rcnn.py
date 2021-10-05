from dataset import ImageLevelSiimCovid19Dataset
from dataset import ResizedImageLevelSiimCovid19Dataset
import utils 

import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
from PIL import Image
import progressbar
import pylibjpeg

import torch
#import torch.nn as nn
#import torch.nn.functional as F


import torch.optim as optim

import torchvision
#import torchvision.transforms as transforms

#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

#from torchvision.transforms import transforms as Tf
#from torchvision.transforms import functional as F
from typing import List, Tuple, Dict, Optional, Any





#number of training epochs
NUM_EPOCHS = 100
BATCH_SIZE = 1

#max size for resizing of input images
#TODO: experiment with diff max_size
IMG_MAX_SIZE = 240

#number of workers for dataloader (set back to zero if keeps getting killed)
NUM_WORKERS = 4

ROOT =  '/media/luisa/Volume/AML/siim-covid19-detection'
CLEAN_TRAIN_PATH = '/media/luisa/Volume/AML/train_image_level_clean_paths.csv'
#alternative path with only images that contain at least on bounding box
#CLEAN_TRAIN_PATH = '/media/luisa/Volume/AML/train_image_level_clean_paths_NOTNA.csv'

RESIZED_TRAIN_PATH = '/media/luisa/Volume/AML/resized_train_image_level_clean_paths.csv'
RESIZED_ROOT = '/media/luisa/Volume/AML/siim-covid19-detection/resized480'


m_save_path = "/media/luisa/Volume/AML/models/fasterrcnn_resnet50_fpn_240_100_epochs_new_anchor"
indices_name = "test_set_fasterrcnn_resnet50_fpn_10_epochs_diffNoBox_v0.csv"
model_name = "fasterrcnn_resnet50_fpn_10_epochs_diffNoBox_v0.pth"
indices_name = "test_set_fasterrcnn_resnet50_fpn_10_epochs_diffNoBox_v0.csv"
model_name = "fasterrcnn_resnet50_fpn_10_epochs_diffNoBox_v0.pth"

#classes for classification (not implemented yet)
CLASSES = ['Negative for Pneumonia',' Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']

#parameters for evaluation
MIN_IOU = 0.5


# model expects input to be list of tensors, our custom dataset therefore requires custom collate_fn function
def collate_fn(batch):
    return tuple(zip(*batch))

class Loss:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0   
    
    
    
if __name__ == "__main__":   
	parser = argparse.ArgumentParser()
	parser.add_argument('--load', type=str, help="path where model is stored")
	args = parser.parse_args()


	print("Save an innocent training process from certain death!")
	print("(Casual reminder to myself that I need to ""sudo swapoff -a"")")
	y_key = input ("press [y] to continue \n")
	
	os.mkdir(m_save_path)
	os.mkdir(os.path.join(m_save_path, 'model'))
	
	model_save_name = os.path.join(m_save_path, 'model', model_name)
	indices_save_name = os.path.join(m_save_path, indices_name)	
	
	# logger	
	logger = logging.getLogger('faster_rcnn_loss')
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler(os.path.join(m_save_path, 'faster_rcnn_loss.log'))
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)



	# train on the GPU if GPU is available
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print('device: ' + str(device))

	#  two classes: background, opacity
	num_classes = 2


	#dataset = ImageLevelSiimCovid19Dataset(ROOT, utils.get_transform(train=True), CLEAN_TRAIN_PATH)
	dataset = ResizedImageLevelSiimCovid19Dataset(RESIZED_ROOT, utils.get_transform(train=True), RESIZED_TRAIN_PATH)
	
	#dataset_test = ImageLevelSiimCovid19Dataset(ROOT, utils.get_transform(train=False))
	
	#TODO: Option to load training indices from somewhere

	# split the random permutation of dataset into train and test set
	indices = torch.randperm(len(dataset)).tolist()

	#save indices of train test split
	
	indices_test = indices[4828:]
	indices_train =  indices[:4828]

	indices_test_df = pd.DataFrame(indices_test)
	indices_test_df.to_csv(indices_save_name)

	# train on 4828 images, keep remaining 1208 for test set, that's about an 80/20 split
	dataset = torch.utils.data.Subset(dataset, indices[:4828])
	#dataset = torch.utils.data.Subset(dataset, indices[:500])
	#dataset_test = torch.utils.data.Subset(dataset_test, indices[4828:])

	# define training and validation data loaders
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)

	#data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

	
	# The model
	
	
	#scales and ratios found by k means
	
	anchor_generator = AnchorGenerator(sizes=((32, 64, 80, 100),), aspect_ratios=((1, 1.5, 2),))
	#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, max_size =IMG_MAX_SIZE )
	
	#train on resized, no max size...max size 480 not possible
	#box_detections_per_img (int): maximum number of detections per image, for all classes.
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img = 3, max_size =IMG_MAX_SIZE, rpn_anchor_generator=anchor_generator )
	
	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	
	# aspect ratio common in this dataset. check sizes after resizing!
	
	'''
	From AnchorGenerator source code:
		    ''The module support computing anchors at multiple sizes and aspect ratios
	    per feature map. This module assumes aspect ratio = height / width for
	    each anchor.

	    sizes and aspect_ratios should have the same number of elements, and it should
	    correspond to the number of feature maps.

	    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
	    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
	    per spatial location for feature map i.

	    Args:
		sizes (Tuple[Tuple[int]]):
		aspect_ratios (Tuple[Tuple[float]]):''
	'''

	
	if args.load:
		print("loading model to continue training")
		path_to_load_from = args.load
		model = torch.load(path_to_load_from)	
	
	
	model.rpn.min_size = 0.0 #needed to prevent NaN loss!!!

	model.to(device)

	# SGD optimizer 
	params = [p for p in model.parameters() if p.requires_grad]
	#optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.01)	
	
	#lower weight_decay and higher lr causes NaN losses
	optimizer = torch.optim.SGD(params, lr=1e-8, momentum=0.9, weight_decay=0.05)

	# learning rate scheduler
	
	#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
	#lr_scheduler = None
	
	#reduce lr only once a plateau is reached
	lr_scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


	
	num_epochs = NUM_EPOCHS
	loss_hist = Loss()
	itr = 1

	#loss_logger = dict()


	torch.save(model, model_save_name)

	print("training started with " + str(NUM_WORKERS) + " workers")
	print()	
	logger.info('training started. images max_size: %s', str(IMG_MAX_SIZE))
	
	for epoch in range(num_epochs):
		loss_hist.reset()
	    
		for images, targets in data_loader:
			images = list(image.to(device) for image in images)
			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
			
			loss_dict = model(images, targets)
			
			losses = sum(loss for loss in loss_dict.values())
			
			loss_value = losses.item()
			
			loss_hist.send(loss_value)
			
			optimizer.zero_grad()
			losses.backward()     
			optimizer.step()
			
			if itr % 50 == 0:
			    
			    iteration_name = 'iteration ' + str(itr)
			    loss_log_msg = iteration_name + ': ' + str(loss_dict)
			    logger.info('loss %s', loss_log_msg)
			    #loss_logger[iteration_name] = loss_dict
			    
			if itr % 500 == 0:
			    print(f"Iteration #{itr} loss: {loss_value}")
			itr += 1
			if lr_scheduler is not None:
			    if epoch > 8:
			        lr_scheduler.step(loss_value)

		print(f"Epoch #{epoch} loss: {loss_hist.value}")  
		torch.save(model, model_save_name)
	    

	   	    

	print()
	print("Training of " + str(num_epochs)+ " completed!")
