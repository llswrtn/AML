from data_wrapper_images import DataWrapperImages
from yolo import Yolo
import torch as T
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import sys
import getopt
import os
from PIL import Image

def test_yolo(device, yolo):
    data = np.zeros((3,448,448))
    data = data.reshape(1, *data.shape)
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
    data[0,5,0,5] = 0.9#ccenterx
    data[0,5,0,6] = 0.3#ccentery
    data[0,5,0,7] = 0.5#w
    data[0,5,0,8] = 0.1#h
    data[0,5,0,9] = 0.75#score

    data[0,6,0,5] = 0.1#ccenterx
    data[0,6,0,6] = 0.3#ccentery
    data[0,6,0,7] = 0.5#w
    data[0,6,0,8] = 0.1#h
    data[0,6,0,9] = 0.74#score

    input_tensor = T.tensor(data, dtype=T.float32, device=device)
    correct_indices, filtered_converted_box_data, filtered_grid_data = yolo.non_max_suppression(0, input_tensor)
    print("correct_indices", correct_indices)
    print("filtered_converted_box_data", filtered_converted_box_data)
    print("filtered_grid_data", filtered_grid_data)
   

def run_tests(device, data_wrapper_images):
    print("running tests...")
    data_wrapper_images.test_get_image_path()
    yolo = Yolo(number_of_classes=4, boxes_per_cell=2).to(device)
    #test_yolo(device, yolo)
    test_non_max_suppression(device, yolo)
    #test_to_converted_box_data(yolo)
    print("tests completed")