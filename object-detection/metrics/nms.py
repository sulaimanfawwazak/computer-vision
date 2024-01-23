#! /usr/bin/env python3
import torch
from iou import intersection_over_union

def nms(
    bboxes,
    iou_threshold,
    threshold,
    box_format="corners"
):
    # bboxes or predictions = [[class_num, prob, x1, y1, x2, y2 (or x, y, w, h)], [], []]
    # class num is to separate different classes (horses, cars, etc)

    """
    Parameters:
        bboxes (list): list of lists containing all bboxex with each bboxes specified as [class_num, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct 
        threshold (float): threshold to remove prediction bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold] # Take the bboxes in the list if the prob is bigger than the IoU threshold
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # Sort the bboxes detected by their prob descendingly
    bboxes_after_nms = []

    # Loops for all bbox in bboxes as long as the list is not empty
    while bboxes:
        chosen_box = bboxes.pop(0) # Take the first bbox in the list

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] # Take the box of the same class (class num)
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold # Take the bbox if the calculated IoU is smaller than the iou_threshold
        ]

        bboxes_after_nms.append(chosen_box) # insert the chosen box to the list

    return bboxes_after_nms