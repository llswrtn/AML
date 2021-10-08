import pickle
import numpy as np
import torch as T

class EpochDataAP():

    def __init__(self, epoch_index):
        self.epoch_index = epoch_index
        self.ap = 0
        self.tp = 0
        self.fp = 0
        self.nd = 0
        self.confusion_matrix = np.zeros((4,4)) 

class EpochLoggerAP():
    """
    The EpochLogger is used to gather detailed loss information about every epoch.
    This allows better and realtime tracking of the training progress.
    """

    def __init__(self, name, num_images):
        self.name = name                    #the name of this logger
        self.num_images = num_images        #the number of images per epoch
        self.num_ground_truth_boxes = 0     #the number of ground truth boxes
        self.epoch_index = -1               #epoch index
        self.list_epoch_data = []

    def start_epoch(self):
        """
        Appends a new EpochDataAP entry.
        """
        self.epoch_index += 1
        self.list_epoch_data.append(EpochDataAP(epoch_index=self.epoch_index))

    def add_epoch_data(self, ap, tp, fp, nd):
        """
        Add epoch data
        """
        epoch_data = self.list_epoch_data[self.epoch_index]
        epoch_data.ap = ap
        epoch_data.tp = tp
        epoch_data.fp = fp
        epoch_data.nd = nd

    def add_prediction(self, predicted_label, ground_truth_label):
        """
        Add epoch data
        """
        epoch_data = self.list_epoch_data[self.epoch_index]
        epoch_data.confusion_matrix[predicted_label, ground_truth_label] += 1

    def print_epoch(self):
        epoch_data = self.list_epoch_data[self.epoch_index]
        print("ap", epoch_data.ap)
        print("tp", epoch_data.tp)
        print("fp", epoch_data.fp)
        print("nd", epoch_data.nd)
        print("confusion_matrix", epoch_data.confusion_matrix)

    def store(self, prefix=""):
        """
        Store the current values.
        """
        file_path = prefix+self.name+".pt"
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    def generate_lists(self):
        """
        Generate converted lists for visualization purposes.
        """
        self.list_ap = []
        self.list_tp = []
        self.list_fp = []
        self.list_nd = []
        self.list_fp_percentage = []
        self.list_predicted_class_0 = []
        self.list_predicted_class_1 = []
        self.list_predicted_class_2 = []
        self.list_predicted_class_3 = []
        self.list_predicted_correct = []
        for i in range(self.epoch_index + 1):       
            tp = self.list_epoch_data[i].tp / self.num_ground_truth_boxes 
            fp = self.list_epoch_data[i].fp / self.num_ground_truth_boxes
            nd = self.list_epoch_data[i].nd / self.num_ground_truth_boxes
            self.list_ap.append(self.list_epoch_data[i].ap)
            self.list_tp.append(tp)
            self.list_fp.append(fp)
            self.list_nd.append(nd)
            print("tp,fp",tp, fp)
            if nd != 0:
                self.list_fp_percentage.append(fp / nd)
            else:
                self.list_fp_percentage.append(0)

            predicted_classes = np.sum(self.list_epoch_data[i].confusion_matrix, axis=1)
            correct_count = np.trace(self.list_epoch_data[i].confusion_matrix)
            self.list_predicted_class_0.append(predicted_classes[0] / self.num_images)
            self.list_predicted_class_1.append(predicted_classes[1] / self.num_images)
            self.list_predicted_class_2.append(predicted_classes[2] / self.num_images)
            self.list_predicted_class_3.append(predicted_classes[3] / self.num_images)
            self.list_predicted_correct.append(correct_count / self.num_images)               