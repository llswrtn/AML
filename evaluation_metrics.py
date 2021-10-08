import numpy as np
#code from Luisa

# first generate tp and fp arrays from predections and gt boxes
# then use tp and fp as input to compute ap, mrec and mpre for PR curve

# input: 

# all_gt_boxes: array where each element represents a gt box in the from [image_id, bounding_box, False]
# GT BOXES IN all_gt_boxes HAVE TO BE SORTED IN DESCENDING ORDER OF THEIR CONFIDENCE SCORE
# example of one element in all_gt_boxes: [252, array([1474.1603,  437.1685, 2245.4863, 1492.4028], dtype=float32), False]

#all_preds: array where each element represents one predicted box in the from [image_id, bounding_box]
# example: 252, array([   0.    , 1601.0254, 2701.    , 1890.7611], dtype=float32), array(1., dtype=float32)]

def generate_tp_fp_arrays (all_gt_boxes_, all_preds ):
	all_gt_boxes = all_gt_boxes_[:]
	all_preds_ = all_preds[:]

	MIN_IOU = 0.5

	#arrays for TP and FP         
	nd = len(all_preds_)
	tp = [0] * nd # creates an array of zeros of size nd
	fp = [0] * nd
	all_tp = 0
	all_fp = 0

	#for each detected objct (detected_box)
	for i in range (len(all_preds)):
	    img_id = all_preds[i][0]
	    pred_box = all_preds[i][1]

	    #count true positives:
	    # set temp_max_iou = -1
	    # set temp_gt_match = -1

	    iou_max = -1
	    gt_match = -1
	    gt_row = -1        
	    
	    #compare the detected_box with all gt_boxes for this image to find max IOU
		#for box in gt boxes of image:
		           
		                                   
	    '''
	    for idx, row in gt_boxes_i.iterrows():
		bb_gt = row['gt_boxes']
		#print(bb_gt)
		bb_pred = pred['pred_boxes']
		used = row['used']

		#print(bb_pred)
		#print()
		#print(used)    
	    '''
	    #gt_boxes_i = all_gt_boxes[1][np.where(all_gt_boxes[0] == img_id)]
	    #used_flag_i = all_gt_boxes[2][np.where(all_gt_boxes[0] == img_id)]
	    
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
		            #print(iou_i)

		            # find max iou for this prediction
		            if iou_i > iou_max:
		                iou_max = iou_i
		                gt_match = pred_box  
		                iou_max_gt_box_idx = j
		                #print(iou_max)


	    if iou_max >= MIN_IOU:
		#print('yes!')
		#set flag used to 'True'
		all_gt_boxes[2][iou_max_gt_box_idx] = True
		#used_flag_i = all_gt_boxes[2][np.where(all_gt_boxes[0] == img_id)]
		#all_gt_boxes_df.at[gt_row.name, 'used'] = True
		
		tp[i] = 1
		all_tp +=1
	    else:
		fp[i] =1
		all_fp +=1    
		
	return tp, fp    




def ap_rec_prec (tp, fp):

	    
	tp_cumsum = np.cumsum(tp, dtype=float)
	rec = tp_cumsum / len(tp)


	fp_cumsum = np.cumsum(fp, dtype=float)
	prec = tp_cumsum /(fp_cumsum + tp_cumsum)

	
	AP, mrec, mprec = voc_ap(rec[:], prec[:])
	
	class_name = 'opacity'

	text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
	print(text)
	return ap, mrec, mprec


# calculates VOC AP as used by the challange

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
 

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
   

    
    return ap, mrec, mpre
    
    

