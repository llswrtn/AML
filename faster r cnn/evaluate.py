# versions of evaluation metrics functions to be used by faster r cnn


import numpy as np
import pandas as pd
import torch

# usage: 
# tp, fp, all_tp, all_fp = generate_tp_fp(data_loader_test)

# ap, mrec, mprec = ap_rec_prec (tp_a, fp_a, fn)

def ap_rec_prec (tp, fp, fn):

   
    tp_cumsum = np.cumsum(tp, dtype=float)
    fn_cumsum= np.cumsum(fn, dtype=float)
    rec = tp_cumsum / (tp_cumsum + fn_cumsum)


    fp_cumsum = np.cumsum(fp, dtype=float)
    prec = tp_cumsum /(fp_cumsum + tp_cumsum)

    
    ap, mrec, mprec = voc_ap(rec[:], prec[:])
    
    class_name = ' opacity'

    text = "{0:.2f}%".format(ap*100) + class_name + " AP "
    print(text)
    return ap, mrec, mprec


def generate_tp_fp (test_dataloader):
    tp = [] #append 1 and the score, later delete score
    fp = [] #append 1 and the score, later delete score
    
    fn = []

    scores = []
    

    

    all_tp = 0
    all_fp = 0

    
    for batch in test_dataloader:
         
        images, targets = batch
        
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        img_id = targets[0]['image_id']
        gt_boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
        #print(gt_boxes)
       

        outputs = model(images)
        #keep = torchvision.ops.nms(outputs[0]['boxes'], outputs[0]['boxes'], NMS_THRESHOLD)
        prd_scores = outputs[0]['scores']
        
        prd_boxes = outputs[0]['boxes']



        #array to flag used gt boxes
        unused = [True]*len(gt_boxes)

        #print(used)


        for i in range(len(prd_boxes)):
        
        # keep max 3 and only if score over conf threshhold
        #print(prd_boxes)
        #for i in range (0, min(3, len(prd_boxes))):
            
            
            if prd_scores[i].cpu().detach().numpy() > CONF_THRESHOLD:
                bb_pred  = prd_boxes[i].cpu().detach().numpy() 
                #print(bb_pred)
                #print(prd_scores[i].cpu().detach().numpy())
                #print()

                iou_max = -1
                gt_match = -1
                gt_row = -1     

                for j in range (len(gt_boxes)):

                        bb_gt = gt_boxes[j]


                        if unused[j] == True:
                            # calculate IOU

                            intersect_box = [max(bb_pred[0],bb_gt[0]), max(bb_pred[1],bb_gt[1]), min(bb_pred[2],bb_gt[2]), min(bb_pred[3],bb_gt[3])]
                            #print(intersect_box)
                            intersect_w = intersect_box[2] - intersect_box[0] + 1
                            intersect_h = intersect_box[3] - intersect_box[1] + 1
                            if intersect_w > 0 and intersect_h >0:


                                union_area = (bb_pred[2] - bb_pred[0] + 1) * (bb_pred[3] - bb_pred[1] + 1) + (bb_gt[2] - bb_gt[0] + 1) * (bb_gt[3] - bb_gt[1] + 1) - intersect_w * intersect_h

                                iou_i = intersect_w * intersect_h / union_area
                                #print(iou_i)

                                # find max iou for this prediction
                                if iou_i > iou_max:
                                    iou_max = iou_i
                                    gt_match = bb_pred
                                    iou_max_gt_box_idx = j
                                    #print(iou_max)


                if iou_max >= MIN_IOU:

                    unused[j] = False

                    tp.append(1) 
                    fp.append(0)
                    fn.append(0)
                    scores.append(prd_scores[i].cpu().detach().numpy() )
                    all_tp +=1
                else:
                    tp.append(0) 
                    fp.append(1) 
                    fn.append(0)
                    scores.append(prd_scores[i].cpu().detach().numpy() )
                    all_fp +=1   
                

                fn[-1] = sum(unused)
                

    tp_ = pd.DataFrame({'tp': tp, 'scores': scores})
    tp_ = tp_.sort_values('scores',  ascending=False)
    tp_ = tp_['tp'].tolist()
    
    fp_ = pd.DataFrame({'fp': fp, 'scores': scores})
    fp_ = fp_.sort_values('scores',  ascending=False)
    fp_ = fp_['fp'].tolist()
    
    return tp_, fp_, all_tp, all_fp, fn
    
    
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

    mrec = np.concatenate([[0.0],rec_,[1.0]])

    mpre = np.concatenate([[0.0],prec_ ,[0.0]])

        
    
    # precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]


    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
   

    return ap, mrec, mpre
    
    

