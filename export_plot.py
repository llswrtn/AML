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

def export_plot(epoch_logger_train, epoch_logger_validate):
    print("export plot, epoch:", epoch_logger_train.epoch_index, ", ", epoch_logger_validate.epoch_index)
    epoch_logger_train.generate_lists()
    epoch_logger_validate.generate_lists()
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="total loss", xlabel="epoch", ylabel="total loss")
    ax.plot(epoch_logger_train.list_total_loss, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_total_loss, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_total_loss.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="L1 (coordinates)", xlabel="epoch", ylabel="L1")
    ax.plot(epoch_logger_train.list_l1, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_l1, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_L1.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="L2 (dimensions)", xlabel="epoch", ylabel="L2")
    ax.plot(epoch_logger_train.list_l2, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_l2, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_L2.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="L3 (confidence obj)", xlabel="epoch", ylabel="L3")
    ax.plot(epoch_logger_train.list_l3, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_l3, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_L3.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="L4 (confidence noobj)", xlabel="epoch", ylabel="L4")
    ax.plot(epoch_logger_train.list_l4, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_l4, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_L4.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="L5 (classification)", xlabel="epoch", ylabel="L5")
    ax.plot(epoch_logger_train.list_l5, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_l5, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_L5.pdf", format="pdf")
    plt.close()

    plt.tight_layout()

if __name__ == "__main__": 
    print("exporting plots")
    epoch_logger_train, epoch_logger_validate = load()
    export_plot(epoch_logger_train, epoch_logger_validate)
