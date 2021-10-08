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

def run_re_validate(device, data_wrapper_images):
    print("run_re_validate")

    ##############################################################################################################
    #
    #       PARAMETERS
    #
    ##############################################################################################################

    batch_size = 64
    continue_validation = False

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
    
    if continue_validation:
        with open("epoch_logger_validate.pt", "rb") as file:
            epoch_logger_validate = pickle.load(file)
    else:
        epoch_logger_validate = EpochLogger(name="epoch_logger_validate", num_images=num_validation_images)

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

    ##############################################################################################################
    #
    #       TRAIN LOOP
    #
    ##############################################################################################################

    T.autograd.set_detect_anomaly(True)
    while epoch_logger_validate.epoch_index < epoch_logger_train.epoch_index:  
        #validate
        run_epoch(False, epoch_logger_validate, batch_size, num_validation_batches, yolo, data_wrapper_images, device)

    
def run_epoch(is_train, epoch_logger, batch_size, num_batches, yolo, data_wrapper_images, device):
    #start of a new epoch  
    epoch_logger.start_epoch()
    print(epoch_logger.name+" start epoch index: ", epoch_logger.epoch_index)
    file_name = str(epoch_logger.epoch_index) + ".pt"
    yolo.load(path="output/"+file_name, device=device)

    yolo.eval()

    T.set_grad_enabled(is_train)

    #loop over all batches
    for batch_index in tqdm(range(num_batches)):
        #get the data of the current train batch
        batch_images, batch_boxes, batch_labels = data_wrapper_images.get_train_or_validation_batch(is_train, batch_index, batch_size, device) 
        #get the results of the yolo network
        forward_result = yolo(batch_images)
        #get everything in one go (loss, class prediction, and boxes)
        total_loss, part_1, part_2, part_3, part_4, part_5, predictions, list_filtered_converted_box_data = yolo.get_batch_loss_and_class_predictions_and_boxes(forward_result, batch_boxes, batch_labels)

        epoch_logger.add_loss(total_loss,part_1,part_2,part_3,part_4,part_5)
    
    #at the end of the epoch, store results
    epoch_logger.store()
    epoch_logger.print_epoch()

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

    run_re_validate(device, data_wrapper_images)