from dataset import ImageLevelSiimCovid19Dataset
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
NUM_EPOCHS = 30
#max size for resizing of input images
#TODO: experiment with diff max_size
IMG_MAX_SIZE = 480
#number of workers for dataloader (set back to zero if keeps getting killed)
NUM_WORKERS = 1

ROOT =  '/media/luisa/Volume/AML/siim-covid19-detection'
CLEAN_TRAIN_PATH = '/media/luisa/Volume/AML/train_image_level_clean_paths.csv'
#alternative path with only images that contain at least on bounding box
#CLEAN_TRAIN_PATH = '/media/luisa/Volume/AML/train_image_level_clean_paths_NOTNA.csv'


m_save_path = "/media/luisa/Volume/AML/models/fasterrcnn_resnet50_fpn_10_epochs_240_v0_continue"
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
	#parser.add_argument('--input', required=True, help="The folder where the serialized data is stored")
	parser.add_argument('--load', type=str, help="path where model is stored")
	args = parser.parse_args()

	sequences_path = config.input
	num_sequences = str(len(os.listdir(sequences_path)))

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


	dataset = ImageLevelSiimCovid19Dataset(ROOT, utils.get_transform(train=True), CLEAN_TRAIN_PATH)
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
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)

	#data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

	# get the model using our helper function
	#model = get_model_instance_segmentation(num_classes) #mask rcnn instance segmentation

	#TODO: option to load model from path given in arg parser
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, max_size =IMG_MAX_SIZE )
	
	if args.load:
		print("loading model to continue training")
		path_to_load_from = args.load
		model = torch.load(path_to_load_from)	
	
	
	model.rpn.min_size = 0.0 #needed to prevent NaN loss!!!

	model.to(device)

	# SGD optimizer 
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=1e-8, momentum=0.9, weight_decay=0.05)
	# and a learning rate scheduler
	#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
	lr_scheduler = None


	
	num_epochs = NUM_EPOCHS
	loss_hist = Loss()
	itr = 1

	#loss_logger = dict()


	torch.save(model, model_save_name)

	print("training started with " + str(NUM_WORKERS) + " workers")
	print()	
	logger.info('training started. images max_size: %s' str(IMG_MAX_SIZE))
	
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
			    lr_scheduler.step()

		print(f"Epoch #{epoch} loss: {loss_hist.value}")  
		torch.save(model, model_save_name)
	    
		#TODO: add eval after each episode
		#evaluation
	    


	    

	print()
	print("Training of " + str(num_epochs)+ " completed!")