import os
import time
import numpy as np
import matplotlib.pyplot as plt
from epoch_logger import *

COLOR_TRAIN = "black"
COLOR_VALIDATE = "green"

def load():
    epoch_logger_train = None
    epoch_logger_validate = None
    with open("epoch_logger_train.pt", "rb") as file:
        epoch_logger_train = pickle.load(file)
    with open("epoch_logger_validate.pt", "rb") as file:
        epoch_logger_validate = pickle.load(file)

    return epoch_logger_train, epoch_logger_validate



def update_plot(epoch_logger_train, epoch_logger_validate, old_epoch_logger_train, old_epoch_logger_validate, fig, axs):
    #check if data changed
    if old_epoch_logger_train != None and old_epoch_logger_validate != None:
        if epoch_logger_train.epoch_index == old_epoch_logger_train.epoch_index and epoch_logger_validate.epoch_index == old_epoch_logger_validate.epoch_index:
            return   

    print("update plot, epoch:", epoch_logger_train.epoch_index, ", ", epoch_logger_validate.epoch_index)
    epoch_logger_train.generate_lists()
    epoch_logger_validate.generate_lists()
    
    for ax in fig.axes:
        ax.cla()
    
    axs[0, 0].set(title="total loss", xlabel="epoch", ylabel="total loss")
    axs[0, 0].plot(epoch_logger_train.list_total_loss, COLOR_TRAIN, label="train")
    axs[0, 0].plot(epoch_logger_validate.list_total_loss, COLOR_VALIDATE, label="validate")
    axs[0, 0].legend(loc="best")

    axs[0, 1].set(title="L1 (coordinates)", xlabel="epoch", ylabel="L1")
    axs[0, 1].plot(epoch_logger_train.list_l1, COLOR_TRAIN, label="train")
    axs[0, 1].plot(epoch_logger_validate.list_l1, COLOR_VALIDATE, label="validate")
    axs[0, 1].legend(loc="best")

    axs[0, 2].set(title="L2 (dimensions)", xlabel="epoch", ylabel="L2")
    axs[0, 2].plot(epoch_logger_train.list_l2, COLOR_TRAIN, label="train")
    axs[0, 2].plot(epoch_logger_validate.list_l2, COLOR_VALIDATE, label="validate")
    axs[0, 2].legend(loc="best")

    axs[1, 0].set(title="L3 (confidence obj)", xlabel="epoch", ylabel="L3")
    axs[1, 0].plot(epoch_logger_train.list_l3, COLOR_TRAIN, label="train")
    axs[1, 0].plot(epoch_logger_validate.list_l3, COLOR_VALIDATE, label="validate")
    axs[1, 0].legend(loc="best")

    axs[1, 1].set(title="L4 (confidence noobj)", xlabel="epoch", ylabel="L4")
    axs[1, 1].plot(epoch_logger_train.list_l3, COLOR_TRAIN, label="train")
    axs[1, 1].plot(epoch_logger_validate.list_l3, COLOR_VALIDATE, label="validate")
    axs[1, 1].legend(loc="best")

    axs[1, 2].set(title="L5 (classification)", xlabel="epoch", ylabel="L5")
    axs[1, 2].plot(epoch_logger_train.list_l3, COLOR_TRAIN, label="train")
    axs[1, 2].plot(epoch_logger_validate.list_l3, COLOR_VALIDATE, label="validate")
    axs[1, 2].legend(loc="best")

    plt.tight_layout()

if __name__ == "__main__": 
    print("hello world")
    old_epoch_logger_train = None
    old_epoch_logger_validate = None

    plt.ion()
    plt.show()
    fig, axs = plt.subplots(2, 3)
    plt.tight_layout()

    while True:
        epoch_logger_train, epoch_logger_validate = load()
        update_plot(epoch_logger_train, epoch_logger_validate, old_epoch_logger_train, old_epoch_logger_validate, fig, axs)
        old_epoch_logger_train = epoch_logger_train
        old_epoch_logger_validate = epoch_logger_validate
        plt.pause(3)
