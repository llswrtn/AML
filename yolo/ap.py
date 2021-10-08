import ast
#import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
from PIL import Image
#import progressbar
import pydicom as dicom
#import pylibjpeg
import torch
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from typing import List, Tuple, Dict, Optional, Any
import pickle

def voc_ap(rec_, prec_):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    #rec.insert(0, 0.0) # insert 0.0 at begining of list
    #rec.append(1.0) # insert 1.0 at end of list
    mrec = np.concatenate([[0.0],rec_,[1.0]])
    
    
    
    #prec.insert(0, 0.0) # insert 0.0 at begining of list
    #prec.append(0.0) # insert 0.0 at end of list
    mpre = np.concatenate([[0.0],prec_ ,[0.0]])
    #print(prec_)
    
    
    
    
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
   
    
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    #for i in range(len(mpre)-2, -1, -1):
        #mpre[i] = max(mpre[i], mpre[i+1])
    #for i in range(mpre.size - 1, 0, -1):
     #   mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
    '''
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    '''

    
    return ap, mrec, mpre

def calculate_ap(all_gt_boxes, all_preds):
    #####################################
    #
    #     PART 1: convert input
    #
    #####################################
    all_preds_df = pd.DataFrame(all_preds, columns=['image_id', 'pred_boxes', 'pred_scores'])    
    all_preds_df = all_preds_df.sort_values('pred_scores',  ascending=False)
    all_gt_boxes_ = np.array(all_gt_boxes).transpose()
    #####################################
    #
    #     PART 2: all_tp, all_fp
    #
    #####################################
    all_gt_boxes = all_gt_boxes_[:]

    MIN_IOU = 0.5

    #arrays for TP and FP         
    nd = len(all_preds_df)
    tp = [0] * nd # creates an array of zeros of size nd
    fp = [0] * nd
    all_tp = 0
    all_fp = 0

    #for each detected objct (detected_box)
    for i in range (len(all_preds_df)):
        img_id = all_preds_df.iloc[i]['image_id']
        pred_box = all_preds_df.iloc[i]['pred_boxes']

        iou_max = -1
        gt_match = -1
        gt_row = -1        
                
        for j in range (len(all_gt_boxes[0])):
            if all_gt_boxes[0][j] == img_id:
                
                bb_gt = all_gt_boxes[1][j]
                used = all_gt_boxes[2][j]
                
                bb_pred = pred_box      
                
                if used == False:
                    # calculate IOU

                    intersect_box = [max(bb_pred[0],bb_gt[0]), max(bb_pred[1],bb_gt[1]), min(bb_pred[2],bb_gt[2]), min(bb_pred[3],bb_gt[3])]
                    #print(intersect_box)
                    intersect_w = intersect_box[2] - intersect_box[0] + 1
                    intersect_h = intersect_box[3] - intersect_box[1] + 1
                    if intersect_w > 0 and intersect_h >0:
                        union_area = (bb_pred[2] - bb_pred[0] + 1) * (bb_pred[3] - bb_pred[1] + 1) + (bb_gt[2] - bb_gt[0] + 1) * (bb_gt[3] - bb_gt[1] + 1) - intersect_w * intersect_h
                        iou_i = intersect_w * intersect_h / union_area

                        # find max iou for this prediction
                        if iou_i > iou_max:
                            iou_max = iou_i
                            gt_match = pred_box  
                            iou_max_gt_box_idx = j
                            #print(iou_max)


        if iou_max >= MIN_IOU:
            all_gt_boxes[2][iou_max_gt_box_idx] = True
            
            tp[i] = 1
            all_tp +=1
        else:
            fp[i] =1
            all_fp +=1    

    #print(all_tp)
    #print(all_fp)


    #####################################
    #
    #     PART 3: sum_AP
    #
    #####################################

    sum_AP = 0
        
    tp_cumsum = np.cumsum(tp, dtype=float)
    rec = tp_cumsum / len(tp)

    fp_cumsum = np.cumsum(fp, dtype=float)
    prec = tp_cumsum /(fp_cumsum + tp_cumsum)
        
    class_name = 'opacity'
    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    sum_AP += ap
    text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)

    #print(text)    

    return [ap, all_tp, all_fp, nd, len(all_gt_boxes[0])]    