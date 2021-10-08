from data_wrapper_images import DataWrapperImages
from yolo import Yolo
from yolo import ARCHITECTURE_DEFAULT
from yolo import ARCHITECTURE_FAST
from yolo import PREDICTION_MAX
from yolo import PREDICTION_MAX_MEAN
from yolo_cell_based import YoloCellBased
from yolo_image_based import YoloImageBased
from plot_boxes import *
from epoch_logger import *
import torch as T
import torchvision
import torch.optim as optim
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import getopt
import os
from tqdm import tqdm
from PIL import Image

def run_validate(device, data_wrapper_images):
    print("run_validate")

    ##############################################################################################################
    #
    #       PARAMETERS
    #
    ##############################################################################################################

    max_num_epochs = 1000
    batch_size = 64
    LEARNING_RATE = 0.0001

    ##############################################################################################################
    #
    #       PREPARE DATA
    #
    ##############################################################################################################

    data_wrapper_images.split_train_validation_test(p_train=0.8, p_validation=0.1, seed=1)
    num_train_batches = data_wrapper_images.get_num_train_batches(batch_size) 
    num_train_images = len(data_wrapper_images.train_indices)
    num_validation_batches = data_wrapper_images.get_num_validation_batches(batch_size) 
    num_validation_images = len(data_wrapper_images.validation_indices)
    num_test_batches = data_wrapper_images.get_num_test_batches(batch_size) 

    ##############################################################################################################
    #
    #       INITIALIZE EPOCH LOGGER
    #
    ##############################################################################################################

    epoch_logger_train = None
    epoch_logger_validate = None
    
    with open("epoch_logger_train.pt", "rb") as file:
        epoch_logger_train = pickle.load(file)
    with open("epoch_logger_validate.pt", "rb") as file:
        epoch_logger_validate = pickle.load(file)

    ##############################################################################################################
    #
    #       INITIALIZE NETWORK
    #
    ##############################################################################################################

    #yolo = YoloCellBased(number_of_classes=4, boxes_per_cell=2).to(device)
    #yolo = YoloImageBased(number_of_classes=4, boxes_per_cell=2).to(device)
    #yolo = YoloCellBased(number_of_classes=4, boxes_per_cell=2, architecture=ARCHITECTURE_FAST).to(device)
    yolo = YoloImageBased(number_of_classes=4, boxes_per_cell=2, architecture=ARCHITECTURE_FAST).to(device)
    yolo.debug = False
    yolo.initialize(device)

    file_name = str(epoch_logger_train.epoch_index) + ".pt"
    yolo.load(path="output/"+file_name, device=device)

    optimizer = optim.Adam(yolo.parameters(), lr=LEARNING_RATE)

    ##############################################################################################################
    #
    #       TRAIN LOOP
    #
    ##############################################################################################################

    #T.autograd.set_detect_anomaly(True)

    yolo.eval()

    T.set_grad_enabled(False)

    all_preds = []
    all_gt_boxes = []
    #loop over all batches
    for batch_index in tqdm(range(num_validation_batches)):
        #zero the gradient since we do not want to use the previous mini batch
        optimizer.zero_grad()
        #get the data of the current train batch
        batch_images, batch_boxes, batch_labels = data_wrapper_images.get_validation_batch(batch_index, batch_size, device) 
        batch_indices = data_wrapper_images.get_validation_batch_indices(batch_index, batch_size)
        #get the results of the yolo network
        forward_result = yolo(batch_images)
        #get everything in one go (loss, class prediction, and boxes)
        total_loss, part_1, part_2, part_3, part_4, part_5, predictions, list_filtered_converted_box_data = yolo.get_batch_loss_and_class_predictions_and_boxes(forward_result, batch_boxes, batch_labels)

        #gather all predicted boxes by iterating over all images of the batch
        for i in range(batch_size):
            image_index = batch_indices[i]
            filtered_converted_box_data = list_filtered_converted_box_data[i]
            #we are interested in non empty box predictions
            if not (filtered_converted_box_data is None):
                #add every box of the prediction
                for j in range(filtered_converted_box_data.shape[0]):
                    box = filtered_converted_box_data[j]
                    box_coords = box[:4].cpu().numpy()
                    box_confidence = box[4].item()
                    entry = [image_index, box_coords, box_confidence]
                    all_preds.append(entry)

            boxes = batch_boxes[i]
            if not (boxes is None):
                #add every box
                for j in range(boxes.shape[0]):
                    box = boxes[j].cpu().numpy()                    
                    entry = [image_index, box, False]
                    all_gt_boxes.append(entry)
        
    with open("all_preds.dat", "wb") as file:
        pickle.dump(all_preds, file)
    with open("all_gt_boxes.dat", "wb") as file:
        pickle.dump(all_gt_boxes, file)
    
if __name__ == "__main__":    
    data_path = "data" 
    data_path_448 = "data_448x448"
    use_cuda = True
    try:
        opts, args = getopt.getopt(sys.argv[1:],"p:d:h",["path=", "downsampled_path=", "help"])
    except getopt.GetoptError:
        print('main.py -p <path of data directory> -d <path of downsampled data directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-p':
            data_path = arg
        if opt == '-d':
            data_path_448 = arg
    data_path += "/"
    data_path_448 += "/"

    #region CUDA
    if use_cuda and T.cuda.is_available():
        print("CUDA ENABLED")
        device = T.device("cuda")
    else:                
        print("CUDA NOT AVAILABLE OR DISABLED")
        device = T.device("cpu")
    #endregion

    data_wrapper_images = DataWrapperImages(data_path, data_path_448)
    data_wrapper_images.load_data_set()

    run_validate(device, data_wrapper_images)
    