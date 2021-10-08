import os
import time
import numpy as np
import matplotlib.pyplot as plt
from epoch_logger import *

#NAME = "2021-10-05_V1" #CHANGE THIS
#NAME = "2021-10-06_V2" #CHANGE THIS
#NAME = None
#NAME = "test" #CHANGE THIS
#NAME = "threshold" #CHANGE THIS
USE_THRESHOLD = False

prefix_results = "results/changed_recall/"

COLOR_TRAIN = "black"
COLOR_VALIDATE = "green"

COLOR_LABEL_0 = "tab:blue"
COLOR_LABEL_1 = "tab:orange"
COLOR_LABEL_2 = "tab:green"
COLOR_LABEL_3 = "tab:red"

epoch_logger_name = "epoch_logger_ap.pt"
if USE_THRESHOLD:
    epoch_logger_name = "epoch_logger_threshold.pt"

def load():
    epoch_logger_ap = None
    with open(prefix_results+epoch_logger_name, "rb") as file:
        epoch_logger_ap = pickle.load(file)

    return epoch_logger_ap

def update_plot(epoch_logger_ap, old_epoch_logger_ap, fig, axs):
    #check if data changed
    if old_epoch_logger_ap != None:
        if epoch_logger_ap.epoch_index == old_epoch_logger_ap.epoch_index:
            return   

    x = np.linspace(0, epoch_logger_ap.epoch_index, epoch_logger_ap.epoch_index + 1)
    if USE_THRESHOLD:
        x = np.linspace(0, 1, epoch_logger_ap.epoch_index + 2)
        x = x[1:]

    print("update plot, epoch:", epoch_logger_ap.epoch_index, "num_boxes:",epoch_logger_ap.num_ground_truth_boxes)
    print("x", x)
    epoch_logger_ap.generate_lists()
    
    for ax in fig.axes:
        ax.cla()
    
    axs[0, 0].set(title="ap", xlabel="epoch", ylabel="")
    axs[0, 0].plot(x, epoch_logger_ap.list_ap, COLOR_VALIDATE, label="validate")
    axs[0, 0].legend(loc="best")

    axs[0, 1].set(title="tp", xlabel="epoch", ylabel="")
    axs[0, 1].plot(x, epoch_logger_ap.list_tp, COLOR_VALIDATE, label="validate")
    axs[0, 1].legend(loc="best")

    axs[0, 2].set(title="class accuracy", xlabel="epoch", ylabel="")
    axs[0, 2].plot(x, epoch_logger_ap.list_predicted_correct, COLOR_VALIDATE, label="validate")
    axs[0, 2].legend(loc="best")


    #axs[1, 0].set(title="nd", xlabel="epoch", ylabel="")
    #axs[1, 0].plot(epoch_logger_ap.list_nd, COLOR_VALIDATE, label="validate")
    #axs[1, 0].legend(loc="best")

    axs[1, 0].set(title="fp / nd", xlabel="epoch", ylabel="")
    axs[1, 0].plot(x, epoch_logger_ap.list_fp_percentage, COLOR_VALIDATE, label="validate")
    axs[1, 0].legend(loc="best")

    axs[1, 1].set(title="fp", xlabel="epoch", ylabel="")
    axs[1, 1].plot(x, epoch_logger_ap.list_fp, COLOR_VALIDATE, label="validate")
    axs[1, 1].legend(loc="best")

    axs[1, 2].set(title="predicted classes", xlabel="epoch", ylabel="")
    #axs[1, 2].plot(epoch_logger_ap.list_predicted_class_0, COLOR_LABEL_0, label="Negative for Pneumonia")
    #axs[1, 2].plot(epoch_logger_ap.list_predicted_class_1, COLOR_LABEL_1, label="Typical Appearance")
    #axs[1, 2].plot(epoch_logger_ap.list_predicted_class_2, COLOR_LABEL_2, label="Indeterminate Appearance")
    #axs[1, 2].plot(epoch_logger_ap.list_predicted_class_3, COLOR_LABEL_3, label="Atypical Appearance")
    axs[1, 2].plot(x, epoch_logger_ap.list_predicted_class_0, COLOR_LABEL_0, label="0")
    axs[1, 2].plot(x, epoch_logger_ap.list_predicted_class_1, COLOR_LABEL_1, label="1")
    axs[1, 2].plot(x, epoch_logger_ap.list_predicted_class_2, COLOR_LABEL_2, label="2")
    axs[1, 2].plot(x, epoch_logger_ap.list_predicted_class_3, COLOR_LABEL_3, label="3")
    axs[1, 2].legend(loc="best")

    plt.tight_layout()

if __name__ == "__main__": 
    print("starting live plot")
    print("prefix_results:", prefix_results)
    
    old_epoch_logger_ap = None

    plt.ion()
    plt.show()
    fig, axs = plt.subplots(2, 3)
    plt.tight_layout()

    while True:
        epoch_logger_ap = load()
        update_plot(epoch_logger_ap, old_epoch_logger_ap, fig, axs)
        old_epoch_logger_ap = epoch_logger_ap
        plt.pause(3)
