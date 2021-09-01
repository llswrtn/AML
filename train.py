# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 05:19:29 2021

@author: User
"""

import torch
import torch.nn as nn
import numpy as np
import nets
import yolo


def train(train_X, test_X, train_y, test_y, device, numChannels, learning_rate = 0.05, num_epochs = 60, batch_size = 256, 
          dropout = 0.2, reproducibility = True, modelName = "simple"):
    #init
    #reproducibility
    if reproducibility == True:
        torch.manual_seed(30)
        torch.cuda.manual_seed(30)
        np.random.seed(30)
        torch.backends.cudnn.deterministic = True
        
    # initialize network
    if (modelName == "simple"):
        model = nets.simpleModel(dropout, train_X.shape[2], numChannels, outputSize = train_y.shape[1])
    elif (modelName == "simpleCNN"):
        model = nets.simpleCNNModel(dropout, train_X.shape[2], numChannels, outputSize = train_y.shape[1])
    else:
        model = yolo.Yolo(train_X.shape[2], numChannels, number_of_classes=6, boxes_per_cell=1, dropout_p=dropout)
    model.to(device)

    #set hyperparameter
    error = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, threshold=0.001, factor=0.5)


    num_Batches = int(train_X.shape[0] / batch_size)
    if train_X.shape[0] % batch_size != 0:
        num_Batches += 1
        
    num_Batches_test = int(test_X.shape[0] / batch_size)
    if test_X.shape[0] % batch_size != 0:
        num_Batches_test += 1
          
    
    #train
    for epoch in range(num_epochs):
        model.train()
        
        train_loss = 0
        #toDo implement 3 metrices below for train and test
        train_meanIOU = 0
        train_shapeAcc = 0
        train_colorAcc = 0
        for i in range(num_Batches):
            #get the correct slice of the data
            cutBatch = (i+1)*batch_size
            if i == num_Batches - 1:
                cutBatch = None
            
            #get the correct batch
            train_y_batch = torch.tensor(train_y[i*batch_size:cutBatch], dtype=torch.float32, device=device)
            train_X_batch = torch.tensor(train_X[i*batch_size:cutBatch], dtype=torch.float32, device=device)
            
            #run data trough model
            optimizer.zero_grad()   
            outputs = model(train_X_batch)
            
            if (modelName == "yolo"):
                for i in range(batch_size):
                    correct_indices, filtered_converted_box_data, filtered_grid_data = yolo.non_max_suppression(i, outputs)
                    
                
                outputs = torch.swapaxes(outputs, 0, 3)
                outputs = torch.cat([outputs[:4,0,0,:], outputs[5:,0,0,:]])
                outputs = torch.swapaxes(outputs, 0, 1)
            loss = error(outputs, train_y_batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss)
        
        train_loss /= num_Batches
    
        # eval
        with torch.no_grad():
            #prepare eval
            model.eval()
            val_loss = 0
            for i in range(num_Batches_test):
                #get the correct slice of the data
                cutBatch = (i+1)*batch_size
                if i == num_Batches_test - 1:
                    cutBatch = None
            
                #get the correct batch
                test_y_batch = torch.from_numpy(test_y[i*batch_size:cutBatch]).float().to(device)
                test_X_batch = torch.from_numpy(test_X[i*batch_size:cutBatch]).float().to(device)

                #run data trough model
                outputs = model(test_X_batch)
                if (modelName == "yolo"):
                    outputs = torch.swapaxes(outputs, 0, 3)
                    outputs = torch.cat([outputs[:4,0,0,:], outputs[5:,0,0,:]])
                    outputs = torch.swapaxes(outputs, 0, 1)
                val_loss += error(outputs, test_y_batch)
            val_loss /= num_Batches_test
            #print progress
            print('Epoch: {} train loss: {} val loss: {}'.format(epoch+1, train_loss, val_loss))
    
    return model



def predict(test_X, model, device, num_objects, num_shapes, num_colors, batch_size, modelName, datasetTyp):
    if (datasetTyp == "simple"):
        pred_y = np.array([], dtype=np.int64).reshape(0,num_objects,4)
    elif (datasetTyp == "multiClasses"):
        pred_y = np.array([], dtype=np.int64).reshape(0,num_objects,10)
    else:
        print("not implemented for X-ray yet")
    num_Batches = int(test_X.shape[0] / batch_size)
    if test_X.shape[0] % batch_size != 0:
        num_Batches += 1
    for i in range(num_Batches):
        cutBatch = (i+1)*batch_size
        if i == num_Batches - 1:
            cutBatch = None
        test_X_Tensor = torch.from_numpy(test_X[i*batch_size:cutBatch]).float().to(device)
        pred_y_tensor = model(test_X_Tensor)
        if (modelName == "yolo"):
            pred_y_tensor = torch.swapaxes(pred_y_tensor, 0, 3)
            pred_y_tensor = torch.cat([pred_y_tensor[:4,0,0,:], pred_y_tensor[5:,0,0,:]])
            pred_y_tensor = torch.swapaxes(pred_y_tensor, 0, 1)
        pred_y_tensor = pred_y_tensor.reshape(len(pred_y_tensor), num_objects, -1)
        pred_y_tensor = pred_y_tensor.detach().cpu().numpy()
        pred_y = np.concatenate((pred_y, pred_y_tensor))
    pred_bboxes = pred_y[..., :4] * test_X.shape[2]
    
    
    if (datasetTyp == "simple"):
        return pred_bboxes, 0, 0
    elif (datasetTyp == "multiClasses"):
        pred_shapes = np.argmax(pred_y[..., 4:4+num_shapes], axis=-1).astype(int)  # take max from probabilities
        pred_colors = np.argmax(pred_y[..., 4+num_shapes:4+num_shapes+num_colors], axis=-1).astype(int)
        return pred_bboxes, pred_shapes, pred_colors
    else:
        print("not implemented for X-ray yet")
        

def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U


def printMeanIOU(pred_bboxes, test_bboxes):
    # Calculate the mean IOU (overlap) between the predicted and expected bounding boxes on the test dataset. 
    summed_IOU = 0.
    for pred_bbox, test_bbox in zip(pred_bboxes.reshape(-1, 4), test_bboxes.reshape(-1, 4)):
        summed_IOU += IOU(pred_bbox, test_bbox)
    mean_IOU = summed_IOU / len(pred_bboxes)
    print("mean IOU: ", mean_IOU)