# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 04:30:46 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cairo

def createSimpleDataSet(num_imgs = 50000, img_size = 8, min_object_size = 1, max_object_size = 4, 
                        num_objects = 1, trainTestSplit = 0.8):
    # Create images with random rectangles and bounding boxes. 
    bboxes = np.zeros((num_imgs, num_objects, 4))
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0

    for i_img in range(num_imgs):
        for i_object in range(num_objects):
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

def createMedicoreDataSet(num_imgs = 50000, img_size = 8, min_object_size = 1, max_object_size = 4, 
                        num_objects = 1, trainTestSplit = 0.8):
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
        
        # TODO: Try no overlap here.
        # Draw random shapes.
        for i_object in range(num_objects):
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
            
            # TODO: Introduce some variation to the colors by adding a small random offset to the rgb values.
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
    #y.shape, np.all(np.argmax(colors_onehot, axis=-1) == colors)
    
    
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
    #train_imgs = imgs[:i]
    #train_bboxes = bboxes[:i]
    
    
    return train_X, test_X, train_y, test_y, test_imgs, test_bboxes, 3, test_shapes, test_colors



def plotImgSimpleDataset(img, bboxes):
    plt.imshow(img.T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img.shape[0], 0, img.shape[0]])
    for bbox in bboxes:
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
    
def plotImgMedicoreDataset(img, bboxes, shapes, colors):
    shape_labels = ['rectangle', 'circle', 'triangle']
    color_labels = ['r', 'g', 'b']
    plt.imshow(img, interpolation='none', origin='lower', extent=[0, img.shape[0], 0, img.shape[0]])
    for bbox, shape, color in zip(bboxes, shapes, colors):
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='k', fc='none'))
        plt.annotate(shape_labels[shape], (bbox[0], bbox[1] + bbox[3] + 0.7), color=color_labels[color], clip_on=False)

