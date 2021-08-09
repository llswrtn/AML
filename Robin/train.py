# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 05:19:29 2021

@author: User
"""

import torch
import torch.nn as nn
import numpy as np
import nets


def train(train_X, test_X, train_y, test_y, numChannels, learning_rate = 0.05, num_epochs = 60, batch_size = 256, 
          dropout = 0.2, reproducibility = True, modelName = "simple"):
    #init
    #reproducibility
    if reproducibility == True:
        torch.manual_seed(30)
        torch.cuda.manual_seed(30)
        np.random.seed(30)
        torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print('Using device:', device)
    
    # initialize network
    if (modelName == "simple"):
        model = nets.simpleModel(dropout, train_X.shape[2], numChannels, outputSize = train_y.shape[1])
    elif (modelName == "simpleCNN"):
        model = nets.simpleCNNModel(dropout, train_X.shape[2], numChannels, outputSize = train_y.shape[1])
    else:
        a = 1###toDo
    model.to(device)

    #set hyperparameter
    error = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, threshold=0.001, factor=0.5)
  
    num_Batches = int(train_X.shape[0] / batch_size)
    if train_X.shape[0] % batch_size != 0:
        num_Batches += 1
            
    #train
    for epoch in range(num_epochs):
        model.train()
        for i in range(num_Batches):
            #get the correct slice of the data
            cutBatch = (i+1)*batch_size
            if i == num_Batches - 1:
                cutBatch = None
            
            #get the correct data for the model type
            train_y_batch = torch.from_numpy(train_y[i*batch_size:cutBatch]).float().to(device)
            train_X_batch = torch.from_numpy(train_X[i*batch_size:cutBatch]).float().to(device)
            #labels = Variable(labels[:,0])
            
            #run data trough model
            optimizer.zero_grad()            
            outputs = model(train_X_batch)
            loss = error(outputs, train_y_batch)
            loss.backward()
            optimizer.step()
            
            
            
        # eval
        with torch.no_grad():
            #prepare eval
            model.eval()
                    
            #get the correct data for model
            test_X_batch = torch.from_numpy(test_X).float().to(device)
            test_y_batch = torch.from_numpy(test_y).float().to(device)

            #run data trough model
            outputs = model(train_X_batch)
            loss = error(outputs, train_y_batch)
            
            #print progress
            print('Epoch: {} val loss: {}'.format(epoch+1, loss))
    return model


def predict(test_X, model, device, num_objects):
    test_X_Tensor = torch.from_numpy(test_X).float().to(device)
    pred_y = model(test_X_Tensor)
    pred_bboxes = pred_y * test_X.shape[2]
    pred_bboxes = pred_bboxes.detach().cpu().numpy()
    pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)
    return pred_bboxes


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