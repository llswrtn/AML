import pickle
import numpy as np
import torch as T

class EpochData():

    def __init__(self, epoch_index):
        self.epoch_index = epoch_index
        self.total_loss = 0
        self.l1 = 0
        self.l2 = 0
        self.l3 = 0
        self.l4 = 0
        self.l5 = 0

class EpochLogger():
    """
    The EpochLogger is used to gather detailed loss information about every epoch.
    This allows better and realtime tracking of the training progress.
    """

    def __init__(self, name, num_images):
        self.name = name                    #the name of this logger
        self.num_images = num_images        #the number of images per epoch
        self.epoch_index = -1               #epoch index
        self.list_epoch_data = []

    def start_epoch(self):
        """
        Appends a new EpochData entry.
        """
        self.epoch_index += 1
        self.list_epoch_data.append(EpochData(epoch_index=self.epoch_index))

    def add_loss(self, total_loss, l1, l2, l3, l4, l5):
        """
        Add loss to the current epoch, can be either batch or individual loss.
        """
        epoch_data = self.list_epoch_data[self.epoch_index]
        epoch_data.total_loss += total_loss.item() if isinstance(total_loss, T.Tensor) else total_loss
        epoch_data.l1 += l1.item() if isinstance(l1, T.Tensor) else l1
        epoch_data.l2 += l2.item() if isinstance(l2, T.Tensor) else l2
        epoch_data.l3 += l3.item() if isinstance(l3, T.Tensor) else l3
        epoch_data.l4 += l4.item() if isinstance(l4, T.Tensor) else l4
        epoch_data.l5 += l5.item() if isinstance(l5, T.Tensor) else l5

    def print_epoch(self):
        epoch_data = self.list_epoch_data[self.epoch_index]
        print("total_loss", epoch_data.total_loss)
        print("l1", epoch_data.l1)
        print("l2", epoch_data.l2)
        print("l3", epoch_data.l3)
        print("l4", epoch_data.l4)
        print("l5", epoch_data.l5)


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
        self.list_total_loss = []
        self.list_l1 = []
        self.list_l2 = []
        self.list_l3 = []
        self.list_l4 = []
        self.list_l5 = []
        for i in range(self.epoch_index + 1):        
            self.list_total_loss.append(self.list_epoch_data[i].total_loss / self.num_images)
            self.list_l1.append(self.list_epoch_data[i].l1 / self.num_images)
            self.list_l2.append(self.list_epoch_data[i].l2 / self.num_images)
            self.list_l3.append(self.list_epoch_data[i].l3 / self.num_images)
            self.list_l4.append(self.list_epoch_data[i].l4 / self.num_images)
            self.list_l5.append(self.list_epoch_data[i].l5 / self.num_images)