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
from epoch_logger_ap import *
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
from ap import *

prefix_results = "results/changed_recall/"
prefix_output = "saved_output/2021-10-05_V1/"

#NAME = "2021-10-05_V1" #CHANGE THIS
#NAME = "2021-10-06_V2" #CHANGE THIS
#NAME = None

def run_gather_ap(device, data_wrapper_images, prefix_results, prefix_output):
    print("run_gather_ap")

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

    data_wrapper_images.split_train_validation_test(p_train=0.8, p_validation=0.2, seed=1)
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
    
    with open(prefix_results+"epoch_logger_train.pt", "rb") as file:
        epoch_logger_train = pickle.load(file)
    
    #with open(prefix_results+"epoch_logger_validate.pt", "rb") as file:
    #    epoch_logger_validate = pickle.load(file)

    epoch_logger_validate = EpochLogger(name="epoch_logger_validate", num_images=num_validation_images)
        
    epoch_logger_ap = EpochLoggerAP(name="epoch_logger_ap", num_images=num_validation_images)

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
        #[ap, all_tp, all_fp]
        run_epoch(epoch_logger_validate, epoch_logger_ap, batch_size, num_validation_batches, yolo, data_wrapper_images, device, prefix_output)

    
def run_epoch(epoch_logger_validate, epoch_logger_ap, batch_size, num_batches, yolo, data_wrapper_images, device, prefix_output):
    #start of a new epoch  
    epoch_logger_validate.start_epoch()
    epoch_logger_ap.start_epoch()
    print(" start epoch index: ", epoch_logger_validate.epoch_index, ", ", epoch_logger_ap.epoch_index)
    file_name = str(epoch_logger_ap.epoch_index) + ".pt"
    yolo.load(path=prefix_output+file_name, device=device)

    yolo.eval()

    T.set_grad_enabled(False)

    all_preds = []
    all_gt_boxes = []
    #loop over all batches
    for batch_index in tqdm(range(num_batches)):
        #get the data of the current train batch
        batch_images, batch_boxes, batch_labels = data_wrapper_images.get_validation_batch(batch_index, batch_size, device) 
        batch_indices = data_wrapper_images.get_validation_batch_indices(batch_index, batch_size)
        #get the results of the yolo network
        forward_result = yolo(batch_images)
        #get everything in one go (loss, class prediction, and boxes)
        total_loss, part_1, part_2, part_3, part_4, part_5, predictions, list_filtered_converted_box_data = yolo.get_batch_loss_and_class_predictions_and_boxes(forward_result, batch_boxes, batch_labels)

        epoch_logger_validate.add_loss(total_loss,part_1,part_2,part_3,part_4,part_5)

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

            predicted_label = T.argmax(predictions[i]).item()
            ground_truth_label = T.argmax(batch_labels[i]).item()
            epoch_logger_ap.add_prediction(predicted_label, ground_truth_label)
    
    #at the end of the epoch, store results
    result = calculate_ap(all_gt_boxes, all_preds)
    epoch_logger_ap.add_epoch_data(result[0], result[1], result[2], result[3])
    epoch_logger_ap.num_ground_truth_boxes = result[4]
    epoch_logger_ap.print_epoch()
    epoch_logger_ap.store(prefix=prefix_results)
    
    epoch_logger_validate.store(prefix=prefix_results)

if __name__ == "__main__":  
    print("prefix_results:", prefix_results)
    print("prefix_output:", prefix_output)

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

    run_gather_ap(device, data_wrapper_images, prefix_results, prefix_output)