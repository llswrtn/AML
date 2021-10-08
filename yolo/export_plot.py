import os
import time
import numpy as np
import matplotlib.pyplot as plt
from epoch_logger import *


export_name = "yolo"




COLOR_TRAIN = "black"
COLOR_VALIDATE = "green"

COLOR_LABEL_0 = "tab:blue"
COLOR_LABEL_1 = "tab:orange"
COLOR_LABEL_2 = "tab:green"
COLOR_LABEL_3 = "tab:red"

def load():
    epoch_logger_train = None
    epoch_logger_validate = None
    epoch_logger_ap = None
    epoch_logger_threshold = None
    with open("epoch_logger_train.pt", "rb") as file:
        epoch_logger_train = pickle.load(file)
    with open("epoch_logger_validate.pt", "rb") as file:
        epoch_logger_validate = pickle.load(file)
    with open("epoch_logger_ap.pt", "rb") as file:
        epoch_logger_ap = pickle.load(file)
    with open("epoch_logger_threshold.pt", "rb") as file:
        epoch_logger_threshold = pickle.load(file)

    return epoch_logger_train, epoch_logger_validate, epoch_logger_ap, epoch_logger_threshold

def export_plot(epoch_logger_train, epoch_logger_validate, epoch_logger_ap, epoch_logger_threshold):
    print("export plot, epoch:", epoch_logger_train.epoch_index, ", ", epoch_logger_validate.epoch_index, ", ", epoch_logger_ap.epoch_index)
    epoch_logger_train.generate_lists()
    epoch_logger_validate.generate_lists()
    epoch_logger_ap.generate_lists()
    epoch_logger_threshold.generate_lists()

    x = np.linspace(0, 1, epoch_logger_threshold.epoch_index + 1)
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="total loss", xlabel="epoch", ylabel="total loss")
    ax.plot(epoch_logger_train.list_total_loss, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_total_loss, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_total_loss.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="L1 (coordinates)", xlabel="epoch", ylabel="L1")
    ax.plot(epoch_logger_train.list_l1, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_l1, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_L1.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="L2 (dimensions)", xlabel="epoch", ylabel="L2")
    ax.plot(epoch_logger_train.list_l2, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_l2, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_L2.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="L3 (confidence obj)", xlabel="epoch", ylabel="L3")
    ax.plot(epoch_logger_train.list_l3, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_l3, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_L3.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="L4 (confidence noobj)", xlabel="epoch", ylabel="L4")
    ax.plot(epoch_logger_train.list_l4, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_l4, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_L4.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="L5 (classification)", xlabel="epoch", ylabel="L5")
    ax.plot(epoch_logger_train.list_l5, COLOR_TRAIN, label="train")
    ax.plot(epoch_logger_validate.list_l5, COLOR_VALIDATE, label="validate")
    ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_L5.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="ap", xlabel="epoch", ylabel="")
    ax.plot(epoch_logger_ap.list_ap, COLOR_VALIDATE)
    #ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_ap.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="tp", xlabel="epoch", ylabel="")
    ax.plot(epoch_logger_ap.list_tp, COLOR_VALIDATE)
    #ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_tp.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="class accuracy", xlabel="epoch", ylabel="")
    ax.plot(epoch_logger_ap.list_predicted_correct, COLOR_VALIDATE)
    #ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_class_accuracy.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="nd", xlabel="epoch", ylabel="")
    ax.plot(epoch_logger_ap.list_nd, COLOR_VALIDATE)
    #ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_nd.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="fp", xlabel="epoch", ylabel="")
    ax.plot(epoch_logger_ap.list_fp, COLOR_VALIDATE)
    #ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_fp.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="predicted classes", xlabel="epoch", ylabel="")
    ax.plot(epoch_logger_ap.list_predicted_class_0, COLOR_LABEL_0, label="0")
    ax.plot(epoch_logger_ap.list_predicted_class_1, COLOR_LABEL_1, label="1")
    ax.plot(epoch_logger_ap.list_predicted_class_2, COLOR_LABEL_2, label="2")
    ax.plot(epoch_logger_ap.list_predicted_class_3, COLOR_LABEL_3, label="3")
    ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_predicted_classes.pdf", format="pdf")
    plt.close()







    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="nd", xlabel="confidence threshold", ylabel="")
    ax.plot(x, epoch_logger_threshold.list_nd, COLOR_VALIDATE)
    #ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_threshold_nd.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="fp", xlabel="confidence threshold", ylabel="")
    ax.plot(x, epoch_logger_threshold.list_fp, COLOR_VALIDATE)
    #ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_threshold_fp.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="tp", xlabel="confidence threshold", ylabel="")
    ax.plot(x, epoch_logger_threshold.list_tp, COLOR_VALIDATE)
    #ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_threshold_tp.pdf", format="pdf")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(title="fp / nd", xlabel="confidence threshold", ylabel="")
    ax.plot(x, epoch_logger_threshold.list_fp_percentage, COLOR_VALIDATE)
    #ax.legend(loc="best")
    plt.savefig(fname="plot_"+export_name+"_threshold_fp_nd.pdf", format="pdf")
    plt.close()

    plt.tight_layout()

if __name__ == "__main__": 
    print("exporting plots")
    epoch_logger_train, epoch_logger_validate, epoch_logger_ap, epoch_logger_threshold = load()
    export_plot(epoch_logger_train, epoch_logger_validate, epoch_logger_ap, epoch_logger_threshold)
