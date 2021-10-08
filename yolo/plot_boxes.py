
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_boxes_and_cells(correct_indices, filtered_converted_box_data, filtered_grid_data):
    fig, ax = plt.subplots()
    
    #create gray rectangles for all active cells
    for i in range(correct_indices.shape[0]):
        min_x = filtered_grid_data[i,2]
        max_x = filtered_grid_data[i,3]
        min_y = filtered_grid_data[i,4]
        max_y = filtered_grid_data[i,5]
        x = (min_x + max_x) / 2
        y = (min_y + max_y) / 2
        w = max_x - min_x
        h = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), w, h, linewidth=1, edgecolor='gray', facecolor='none')
        ax.add_patch(rect)

    #create red rectangles for all active boxes
    for i in range(correct_indices.shape[0]):
        min_x = filtered_converted_box_data[i,0]
        min_y = filtered_converted_box_data[i,1]
        max_x = filtered_converted_box_data[i,2]
        max_y = filtered_converted_box_data[i,3]
        x = (min_x + max_x) / 2
        y = (min_y + max_y) / 2
        w = max_x - min_x
        h = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def plot_responsible_and_ground_truth(responsible_indices, converted_box_data, ground_truth_boxes):
    fig, ax = plt.subplots()
    
    #create green rectangles for all ground truth boxes
    print(ground_truth_boxes.shape[0])
    for i in range(ground_truth_boxes.shape[0]):
        print("draw ground truth", ground_truth_boxes[i])
        min_x = ground_truth_boxes[i,0]
        min_y = ground_truth_boxes[i,1]
        max_x = ground_truth_boxes[i,2]
        max_y = ground_truth_boxes[i,3]
        x = (min_x + max_x) / 2
        y = (min_y + max_y) / 2
        w = max_x - min_x
        h = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), w, h, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    #create red rectangles for all responsible boxes
    for i, index in enumerate(responsible_indices):
        print("draw box", converted_box_data[i])
        min_x = converted_box_data[index,0]
        min_y = converted_box_data[index,1]
        max_x = converted_box_data[index,2]
        max_y = converted_box_data[index,3]
        x = (min_x + max_x) / 2
        y = (min_y + max_y) / 2
        w = max_x - min_x
        h = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def plot_intersected_cells_and_ground_truth(intersected_cells_mask, grid_data, ground_truth_boxes):
    fig, ax = plt.subplots()
    
    #create green rectangles for all ground truth boxes
    print(ground_truth_boxes.shape[0])
    for i in range(ground_truth_boxes.shape[0]):
        print("draw ground truth", ground_truth_boxes[i])
        min_x = ground_truth_boxes[i,0]
        min_y = ground_truth_boxes[i,1]
        max_x = ground_truth_boxes[i,2]
        max_y = ground_truth_boxes[i,3]
        x = (min_x + max_x) / 2
        y = (min_y + max_y) / 2
        w = max_x - min_x
        h = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), w, h, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    #create gray rectangles for all intersected cells
    for i in range(intersected_cells_mask.shape[0]):
        draw = False
        for j in range(intersected_cells_mask.shape[1]):
            if intersected_cells_mask[i,j]:
                draw = True
                break
        if not draw:
            continue
        print("draw intersected cell", grid_data[i])
        min_x = grid_data[i,2]
        max_x = grid_data[i,3]
        min_y = grid_data[i,4]
        max_y = grid_data[i,5]
        x = (min_x + max_x) / 2
        y = (min_y + max_y) / 2
        w = max_x - min_x
        h = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), w, h, linewidth=1, edgecolor='gray', facecolor='none')
        ax.add_patch(rect)

    plt.show()