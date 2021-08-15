from data_wrapper_images import DataWrapperImages
from yolo import Yolo
from yolo import ARCHITECTURE_DEFAULT
from yolo import ARCHITECTURE_FAST
from yolo import PREDICTION_MAX
from yolo import PREDICTION_MAX_MEAN
from yolo_cell_based import YoloCellBased
from yolo_image_based import YoloImageBased
from plot_boxes import *
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


def test_yolo(device, yolo):
    #data = np.zeros((3,448,448))
    #data = data.reshape(1, *data.shape)
    data = np.random.rand(2,1,448,448)
    input_tensor = T.tensor(data, dtype=T.float32, device=device)
    output_tensor = yolo(input_tensor)
    print("output_tensor", output_tensor)
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

    forward_result = T.tensor(data, dtype=T.float32, device=device)
    converted_box_data = yolo.prepare_data(0, forward_result)
    print("converted_box_data", converted_box_data)
    correct_indices, filtered_converted_box_data, filtered_grid_data = yolo.non_max_suppression(converted_box_data)
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

def test_get_responsible_indices(device, yolo):
    np.random.seed(1)
    data = np.random.rand(64,yolo.S,yolo.S,yolo.values_per_cell)

    ground_truth_boxes = np.array([
        [0.2, 0.2, 0.4, 0.4], 
        [0.6, 0.6, 0.8, 0.8]])
    ground_truth_boxes = T.tensor(ground_truth_boxes, dtype=T.float32, device=device)

    #simulate forward
    dummy_forward_result = T.tensor(data, dtype=T.float32, device=device)
    #convert result of forward
    converted_box_data = yolo.prepare_data(0, dummy_forward_result)
    #get responsible indices
    responsible_indices, responsible_indices_1, responsible_indices_any_1, responsible_indices_noobj_1 = yolo.get_responsible_indices(converted_box_data, ground_truth_boxes)
    #plot
    plot_responsible_and_ground_truth(responsible_indices, converted_box_data, ground_truth_boxes)

def test_get_intersected_cells(device, yolo):
    ground_truth_boxes = np.array([
        [0.2, 0.2, 0.4, 0.4], 
        [0.6, 0.6, 0.8, 0.8]])
    ground_truth_boxes = T.tensor(ground_truth_boxes, dtype=T.float32, device=device)

    intersected_cells_mask, intersected_cells_1 = yolo.get_intersected_cells(ground_truth_boxes)
    #plot
    plot_intersected_cells_and_ground_truth(intersected_cells_mask, yolo.grid_data, ground_truth_boxes)

def test_get_responsible_cells(device, yolo):
    ground_truth_boxes = np.array([
        [0.2, 0.2, 0.4, 0.4], 
        [0.6, 0.6, 0.8, 0.8]])
    ground_truth_boxes = T.tensor(ground_truth_boxes, dtype=T.float32, device=device)

    responsible_cells_mask, responsible_cells_1, responsible_cells_index_list = yolo.get_responsible_cells(ground_truth_boxes)
    #plot
    plot_intersected_cells_and_ground_truth(responsible_cells_mask, yolo.grid_data, ground_truth_boxes)

def test_loss(device, yolo, data_wrapper_images):
    np.random.seed(1)
    """
    data = np.random.rand(64,yolo.S,yolo.S,yolo.values_per_cell)

    ground_truth = np.array([
        [0.2, 0.2, 0.4, 0.4], 
        [0.6, 0.6, 0.8, 0.8]])
    ground_truth = T.tensor(ground_truth, dtype=T.float32, device=device)

    ground_truth_label = np.array([0, 0, 1, 0])
    ground_truth_label = T.tensor(ground_truth_label, dtype=T.float32, device=device)

    #simulate forward
    dummy_forward_result = T.tensor(data, dtype=T.float32, device=device)

    #random forward
    data = np.random.rand(2,1,448,448)
    input_tensor = T.tensor(data, dtype=T.float32, device=device)
    forward_result = yolo(input_tensor)

    #convert result of forward
    converted_box_data = yolo.prepare_data(0, forward_result)
    #call loss function for one image
    yolo.get_loss(converted_box_data, ground_truth, ground_truth_label)
    """
    
    data = np.random.rand(2,1,448,448)
    input_tensor = T.tensor(data, dtype=T.float32, device=device)
    forward_result = yolo(input_tensor)
    print("forward_result", forward_result)

    data_wrapper_images.split_train_validation_test(p_train=0.8, p_validation=0.1, seed=1)
    batch_size = 2
    num_test_batches = data_wrapper_images.get_num_test_batches(batch_size) 
    print("num_test_batches", num_test_batches)
    batch_images, batch_boxes, batch_labels = data_wrapper_images.get_test_batch(0, batch_size, device=device) 
    print("batch_images.shape", batch_images.shape)
    print("len(batch_boxes)", len(batch_boxes))
    print("batch_labels.shape", batch_labels.shape)
    forward_result = yolo(batch_images)
    #call loss function for batch
    loss = yolo.get_batch_loss(forward_result, batch_boxes, batch_labels)
    #call prediction function for batch
    predictions = yolo.get_batch_class_predictions(forward_result)
    print("predictions", predictions)
    print("batch_labels", batch_labels)

    total_loss, predictions, list_filtered_converted_box_data = yolo.get_batch_loss_and_class_predictions_and_boxes(forward_result, batch_boxes, batch_labels)
    print("total_loss", total_loss)
    print("predictions", predictions)
    print("list_filtered_converted_box_data", list_filtered_converted_box_data)


def test_data(data_wrapper_images, device):    
    print("test_data")
    data_wrapper_images.split_train_validation_test(p_train=0.8, p_validation=0.1, seed=1)

    batch_size = 8
    num_test_batches = data_wrapper_images.get_num_test_batches(batch_size) 
    print("num_test_batches", num_test_batches)
    batch_images, batch_boxes, batch_labels = data_wrapper_images.get_test_batch(0, batch_size, device) 
    print("batch_images", batch_images)
    print("batch_boxes", batch_boxes)
    print("batch_labels", batch_labels)
    print("batch_images.shape", batch_images.shape)
    print("len(batch_boxes)", len(batch_boxes))
    print("batch_labels.shape", batch_labels.shape)
    #print(data_wrapper_images.image_size_list)
    #test_batch = data_wrapper_images.get_test_batch(1, batch_size) 
    #print("test_batch", test_batch)
    #test_batch = data_wrapper_images.get_test_batch(78, batch_size) 
    #print("test_batch", test_batch)

def run_tests(device, data_wrapper_images):
    print("running tests...")
    test_data(data_wrapper_images, device)
    data_wrapper_images.test_get_image_path()
    #sys.exit(0)
    #test_iou(device)
    #yolo = YoloCellBased(number_of_classes=4, boxes_per_cell=2).to(device)
    #yolo = YoloImageBased(number_of_classes=4, boxes_per_cell=2).to(device)
    #yolo = YoloCellBased(number_of_classes=4, boxes_per_cell=2, architecture=ARCHITECTURE_FAST).to(device)
    yolo = YoloImageBased(number_of_classes=4, boxes_per_cell=2, architecture=ARCHITECTURE_FAST).to(device)
    yolo.initialize(device)
    #test_get_intersected_cells(device, yolo)
    #test_get_responsible_cells(device, yolo)
    #test_get_responsible_indices(device, yolo)
    test_loss(device, yolo, data_wrapper_images)
    #test_yolo(device, yolo)
    #test_non_max_suppression(device, yolo)
    #test_to_converted_box_data(yolo)
    print("tests completed")