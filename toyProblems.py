# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 04:30:46 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cairo

import train


def createDataset(num_imgs, img_size, min_object_size, max_object_size, num_objects, trainTestSplit, 
                  datasetTyp, randomNumObj):
    if (datasetTyp == "simple"):
        return createSimpleDataSet(num_imgs, img_size, min_object_size, max_object_size, num_objects, 
                                   trainTestSplit, randomNumObj)
    elif (datasetTyp == "multiClasses"):
        return createMultiClassesDataSet(num_imgs, img_size, min_object_size, max_object_size, num_objects, 
                                   trainTestSplit, randomNumObj)
    else:
        print("X-ray not implemented yet")

def createSimpleDataSet(num_imgs, img_size, min_object_size, max_object_size, num_objects, 
                        trainTestSplit, randomNumObj):
    # Create images with random rectangles and bounding boxes. 
    bboxes = np.zeros((num_imgs, num_objects, 4))
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0

    for i_img in range(num_imgs):
        if (randomNumObj):
            num_obj_in_image = np.random.randint(num_objects+1)
        else:
            num_obj_in_image = num_objects
        for i_object in range(num_obj_in_image):
            w, h = np.random.randint(min_object_size, max_object_size, size=2)
            x = np.random.randint(0, img_size - w)
            y = np.random.randint(0, img_size - h)
            imgs[i_img, x:x+w, y:y+h] = 1.  # set rectangle to 1
            bboxes[i_img, i_object] = [x, y, w, h]
    
    
    # Reshape and normalize the image data to mean 0 and std 1. 
    X = (imgs - np.mean(imgs)) / np.std(imgs)
    #print(X.shape, np.mean(X), np.std(X))
    
    # Normalize bboxes so that all values are between 0 and 1.
    y = bboxes.reshape(num_imgs, -1) / img_size
    #print(y.shape, np.mean(y), np.std(y))
    
    # Split training and test.
    i = int(trainTestSplit * num_imgs)
    train_X = X[:i]
    test_X = X[i:]
    train_y = y[:i]
    test_y = y[i:]
    test_imgs = imgs[i:]
    test_bboxes = bboxes[i:]
    #train_imgs = imgs[:i]
    #train_bboxes = bboxes[:i]
    train_X = np.expand_dims(train_X, axis=1)
    test_X = np.expand_dims(test_X, axis=1)

    return train_X, test_X, train_y, test_y, test_imgs, test_bboxes, 1, 0, 0

def createMultiClassesDataSet(num_imgs, img_size, min_object_size, max_object_size, 
                        num_objects, trainTestSplit, randomNumObj):
    bboxes = np.zeros((num_imgs, num_objects, 4))
    imgs = np.zeros((num_imgs, img_size, img_size, 4), dtype=np.uint8)  # format: BGRA
    shapes = np.zeros((num_imgs, num_objects), dtype=int)
    num_shapes = 3
    shape_labels = ['rectangle', 'circle', 'triangle']
    colors = np.zeros((num_imgs, num_objects), dtype=int)
    num_colors = 3
    color_labels = ['r', 'g', 'b']
    
    for i_img in range(num_imgs):
        surface = cairo.ImageSurface.create_for_data(imgs[i_img], cairo.FORMAT_ARGB32, img_size, img_size)
        cr = cairo.Context(surface)
    
        # Fill background white.
        cr.set_source_rgb(1, 1, 1)
        cr.paint()
        
        # Draw random shapes.
        if (randomNumObj):
            num_obj_in_image = np.random.randint(num_objects+1)
        else:
            num_obj_in_image = num_objects
        for i_object in range(num_obj_in_image):
            shape = np.random.randint(num_shapes)
            shapes[i_img, i_object] = shape
            if shape == 0:  # rectangle
                w, h = np.random.randint(min_object_size, max_object_size, size=2)
                x = np.random.randint(0, img_size - w)
                y = np.random.randint(0, img_size - h)
                bboxes[i_img, i_object] = [x, y, w, h]
                cr.rectangle(x, y, w, h)            
            elif shape == 1:  # circle   
                r = 0.5 * np.random.randint(min_object_size, max_object_size)
                x = np.random.randint(r, img_size - r)
                y = np.random.randint(r, img_size - r)
                bboxes[i_img, i_object] = [x - r, y - r, 2 * r, 2 * r]
                cr.arc(x, y, r, 0, 2*np.pi)
            elif shape == 2:  # triangle
                w, h = np.random.randint(min_object_size, max_object_size, size=2)
                x = np.random.randint(0, img_size - w)
                y = np.random.randint(0, img_size - h)
                bboxes[i_img, i_object] = [x, y, w, h]
                cr.move_to(x, y)
                cr.line_to(x+w, y)
                cr.line_to(x+w, y+h)
                cr.line_to(x, y)
                cr.close_path()
            
            # add noise to the colors
            color = np.random.randint(num_colors)
            colors[i_img, i_object] = color
            max_offset = 0.3
            r_offset, g_offset, b_offset = max_offset * 2. * (np.random.rand(3) - 0.5)
            if color == 0:
                cr.set_source_rgb(1-max_offset+r_offset, 0+g_offset, 0+b_offset)
            elif color == 1:
                cr.set_source_rgb(0+r_offset, 1-max_offset+g_offset, 0+b_offset)
            elif color == 2:
                cr.set_source_rgb(0+r_offset, 0-max_offset+g_offset, 1+b_offset)
            cr.fill()
            
    imgs = imgs[..., 2::-1]  # is BGRA, convert to RGB
    
    
    # Reshape and normalize the image data to mean 0 and std 1. 
    X = (imgs - 128.) / 255.
    X.shape, np.mean(X), np.std(X)


    #onehot labels
    colors_onehot = np.zeros((num_imgs, num_objects, num_colors))
    for i_img in range(num_imgs):
        for i_object in range(num_objects):
            colors_onehot[i_img, i_object, colors[i_img, i_object]] = 1
    
    shapes_onehot = np.zeros((num_imgs, num_objects, num_shapes))
    for i_img in range(num_imgs):
        for i_object in range(num_objects):
            shapes_onehot[i_img, i_object, shapes[i_img, i_object]] = 1
            
    y = np.concatenate([bboxes / img_size, shapes_onehot, colors_onehot], axis=-1).reshape(num_imgs, -1)
    
    
    # Split training and test.
    i = int(trainTestSplit * num_imgs)
    train_X = X[:i]
    test_X = X[i:]
    train_y = y[:i]
    test_y = y[i:]
    test_imgs = imgs[i:]
    test_bboxes = bboxes[i:]
    test_shapes = shapes[i:]
    test_colors = colors[i:]
    
    train_X = np.swapaxes(train_X, 1,3)
    train_X = np.swapaxes(train_X, 2,3)
    test_X = np.swapaxes(test_X, 1,3)
    test_X = np.swapaxes(test_X, 2,3)
    #train_imgs = imgs[:i]
    #train_bboxes = bboxes[:i]
    
    return train_X, test_X, train_y, test_y, test_imgs, test_bboxes, 3, test_shapes, test_colors



def plotImgs(imgs, bboxes, shapes, colors, datasetTyp, idxImagesToPlot = [0], pred_bboxes = None, 
             pred = False):
    if (datasetTyp == "simple"):
        return plotImgSimpleDataset(imgs, bboxes, idxImagesToPlot, pred_bboxes, pred)
    elif (datasetTyp == "multiClasses"):
        return plotImgMultiClassesDataset(imgs, bboxes, shapes, colors, idxImagesToPlot, pred_bboxes, pred)
    else:
        print("X-ray not implemented yet")

def plotImgSimpleDataset(imgs, bboxes, idxImagesToPlot, pred_bboxes = None, pred = False):
    plt.figure(figsize=(16, 8))
    for i_subplot,idx in enumerate(idxImagesToPlot):
        plt.subplot(1, len(idxImagesToPlot), i_subplot+1)
        plt.imshow(imgs[idx].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, imgs.shape[2], 0, imgs.shape[2]])
        if (pred):
            for pred_bbox, exp_bbox in zip(pred_bboxes[idx], bboxes[idx]):
                if (pred_bbox[2] == 0.0 and pred_bbox[3] == 0.0):
                    plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r', fc='none'))
                    plt.annotate('IOU: {:.2f}'.format(train.IOU(pred_bbox, exp_bbox)), (pred_bbox[0], pred_bbox[1]+pred_bbox[3]+0.2), color='r')
        else:
            for bbox in bboxes[idx]:
                if (bbox[2] == 0.0 and bbox[3] == 0.0):
                    continue
                plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
       
                 
            
    
def plotImgMultiClassesDataset(imgs, bboxes, shapes, colors, idxImagesToPlot, pred_bboxes = None, 
                               pred = False):
    shape_labels = ['rectangle', 'circle', 'triangle']
    color_labels = ['r', 'g', 'b']
    plt.figure(figsize=(16, 8))
    for i_subplot,idx in enumerate(idxImagesToPlot):
        plt.subplot(2, 4, i_subplot+1)
        plt.imshow(imgs[idx], interpolation='none', origin='lower', extent=[0, imgs.shape[2], 0, imgs.shape[2]])
        if (pred):
            for bbox, shape, color, exp_bbox in zip(pred_bboxes[idx], shapes[idx], colors[idx], bboxes[idx]):
                if (bbox[2] == 0.0 and bbox[3] == 0.0):
                    continue
                plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='k', fc='none'))
                plt.annotate(shape_labels[shape], (bbox[0], bbox[1] + bbox[3] + 3.7), color=color_labels[color], clip_on=False, bbox={'fc': 'w', 'ec': 'none', 'pad': 1, 'alpha': 0.6})
                plt.annotate('IOU: {:.2f}'.format(train.IOU(bbox, exp_bbox)), (bbox[0], bbox[1]+bbox[3]+0.2))
        else:
            for bbox, shape, color in zip(bboxes[idx], shapes[idx], colors[idx]):
                if (bbox[2] == 0.0 and bbox[3] == 0.0):
                    continue
                plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='k', fc='none'))
                plt.annotate(shape_labels[shape], (bbox[0], bbox[1] + bbox[3] + 0.7), color=color_labels[color], clip_on=False)
        
            
            