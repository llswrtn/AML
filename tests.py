from data_wrapper_images import DataWrapperImages
from yolo import Yolo
import torch as T
import torchvision
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import getopt
import os
from PIL import Image

def plot_boxes_and_cells(correct_indices, filtered_converted_box_data, filtered_grid_data):
    fig, ax = plt.subplots()
    
    #create gray rectangles for all active cells
    for i in range(correct_indices.shape[0]):
        min_x = filtered_grid_data[i,2]
        max_x = filtered_grid_data[i,3]
        min_y = filtered_grid_data[i,4]
        max_y = filtered_grid_data[i,5]
        x = (min_x + max_x) / 2
        y = (min_y + max_y) / 2
        w = max_x - min_x
        h = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), w, h, linewidth=1, edgecolor='gray', facecolor='none')
        ax.add_patch(rect)

    #create red rectangles for all active boxes
    for i in range(correct_indices.shape[0]):
        min_x = filtered_converted_box_data[i,0]
        min_y = filtered_converted_box_data[i,1]
        max_x = filtered_converted_box_data[i,2]
        max_y = filtered_converted_box_data[i,3]
        x = (min_x + max_x) / 2
        y = (min_y + max_y) / 2
        w = max_x - min_x
        h = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def test_yolo(device, yolo):
    data = np.zeros((3,448,448))
    data = data.reshape(1, *data.shape)
    data = np.random.rand(2,1,448,448)
    input_tensor = T.tensor(data, dtype=T.float32, device=device)
    output_tensor = yolo(input_tensor)
    #print(output_tensor)
    print(output_tensor.shape)
    #yolo.save("save.pt")#--> 1GB

def test_to_converted_box_data(yolo):
    separate_box_data = T.zeros((yolo.S*yolo.S*yolo.B, 5*yolo.B+yolo.C)) 
    separate_box_data[11,0] = 0.5
    separate_box_data[11,1] = 0.75
    separate_box_data[11,2] = 0.1
    separate_box_data[11,3] = 0.2

    separate_box_data[22,0] = 0.25
    separate_box_data[22,1] = 0.9
    separate_box_data[22,2] = 0.2
    separate_box_data[22,3] = 0.25

    converted_box_data = yolo.to_converted_box_data(separate_box_data)
    print("stacked_grid_data", yolo.stacked_grid_data[11])
    print("converted_box_data", converted_box_data[11])
    print("stacked_grid_data", yolo.stacked_grid_data[22])
    print("converted_box_data", converted_box_data[22])

def test_non_max_suppression(device, yolo):
    data = np.zeros((64,yolo.S,yolo.S,yolo.values_per_cell))
    #generate two boxes in the same cell
    data[0,0,0,0] = 0.5#ccenterx
    data[0,0,0,1] = 0.5#ccentery
    data[0,0,0,2] = 0.2#w
    data[0,0,0,3] = 0.1#h
    data[0,0,0,4] = 0.8#score

    data[0,0,0,5] = 0.5#ccenterx
    data[0,0,0,6] = 0.5#ccentery
    data[0,0,0,7] = 0.25#w
    data[0,0,0,8] = 0.1#h
    data[0,0,0,9] = 0.9#score

    #generate two overlapping boxes in neighboring cells
    data[0,4,5,5] = 0.9#ccenterx
    data[0,4,5,6] = 0.3#ccentery
    data[0,4,5,7] = 0.3#w
    data[0,4,5,8] = 0.1#h
    data[0,4,5,9] = 0.75#score

    data[0,5,5,5] = 0.1#ccenterx
    data[0,5,5,6] = 0.3#ccentery
    data[0,5,5,7] = 0.3#w
    data[0,5,5,8] = 0.1#h
    data[0,5,5,9] = 0.74#score

    #generate two overlapping boxes in neighboring cells
    data[0,4,3,5] = 0.8#ccenterx
    data[0,4,3,6] = 0.4#ccentery
    data[0,4,3,7] = 0.2#w
    data[0,4,3,8] = 0.1#h
    data[0,4,3,9] = 0.75#score

    data[0,5,3,5] = 0.3#ccenterx
    data[0,5,3,6] = 0.7#ccentery
    data[0,5,3,7] = 0.18#w
    data[0,5,3,8] = 0.18#h
    data[0,5,3,9] = 0.74#score

    input_tensor = T.tensor(data, dtype=T.float32, device=device)
    correct_indices, filtered_converted_box_data, filtered_grid_data = yolo.non_max_suppression(0, input_tensor)
    print("correct_indices", correct_indices)
    print("filtered_converted_box_data", filtered_converted_box_data)
    print("filtered_grid_data", filtered_grid_data)
    plot_boxes_and_cells(correct_indices, filtered_converted_box_data, filtered_grid_data)
   
def test_iou(device):
    #predicted boxes
    boxes1 = np.array([
        [0.5, 0.5, 0.6, 0.6],
        [0.0, 0.0, 0.2, 0.1], 
        [0.0, 0.0, 0.3, 0.1],
        [0.8, 0.8, 0.91, 0.91]])

    #ground truth boxes
    boxes2 = np.array([
        [0.0, 0.0, 0.2, 0.1], 
        [0.9, 0.9, 1.0, 1.0]])

    boxes1 = T.tensor(boxes1, dtype=T.float32, device=device)
    boxes2 = T.tensor(boxes2, dtype=T.float32, device=device)
    #(x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2
    iou = torchvision.ops.box_iou(boxes1, boxes2)
    responsible_indices = T.argmax(iou, dim=0)
    print("iou", iou)
    print("responsible_indices", responsible_indices)

def run_tests(device, data_wrapper_images):
    print("running tests...")
    data_wrapper_images.test_get_image_path()
    test_iou(device)
    #yolo = Yolo(number_of_classes=4, boxes_per_cell=2).to(device)
    #test_yolo(device, yolo)
    #test_non_max_suppression(device, yolo)
    #test_to_converted_box_data(yolo)
    print("tests completed")