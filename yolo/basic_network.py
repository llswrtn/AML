import os
import sys
import pickle
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from plot_boxes import *

class BasicNetwork(nn.Module):
    """
    Basic functionality like saving and loading
    """

    def __init__(self):   
        """
        :param input_dims: TODO
        """   
        super(BasicNetwork, self).__init__()   
        self.debug = True
        print("init BasicNetwork")     

    def print_debug(self, text, value):
        if(self.debug):
            print(text, value)

    def forward(self, state):
        pass

    def get_conv_out_size(self, input_size, kernel_size, stride, padding):
        """
        see https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer
        """
        return math.floor((input_size - kernel_size + 2 * padding ) / stride + 1)

    def copy_from(self, other):
        """
        Copies the state_dict of the provided other network.

        :param other: the network whose state_dict should be copied into this network
        """
        theta = other.state_dict()
        self.load_state_dict(theta)

    def save(self, path):
        """
        saves the network to the specified path
        """
        print(f"save network to: {path}")
        try:
            with open(path, "wb") as file:
                T.save(self.state_dict(), file)
        except Exception as e:
            print("error saving network")
            print(e)

    def load(self, path, device):
        """
        loads the network to the specified device
        """
        print(f"load network: {path}")
        if not os.path.isfile(path):
            print(f"could not find file: {path}")
            return

        try:
            print("try loading network...")
            with open(path, "rb") as file:
                theta = T.load(file, map_location=device)
                self.load_state_dict(theta)  
            print("loaded network successfully")           
        except Exception as e:
            print("error loading network")
            print(e)