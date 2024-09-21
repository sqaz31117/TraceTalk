import numpy as np
import os

img_w, img_h = 2704, 1520

def calculate_iou(box1, box2):
    # Calculate the intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the areas of the boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IOU
    iou = intersection_area / union_area
    return iou

def xywh2xyxy(box):
    cx = float(box[0])
    cy = float(box[1])
    w = float(box[2])
    h = float(box[3])

    x_left = int(cx - w/2.)
    x_right = int(cx + w/2.)
    y_top = int(cy - h/2.)
    y_bottom = int(cy + h/2.)
    return x_left, y_top, x_right, y_bottom

def label2coord(label, height_image, width_image):
    category      =       label[0]
    x_center_bbox = float(label[1])
    y_center_bbox = float(label[2])
    width_bbox    = float(label[3])
    height_bbox   = float(label[4])
    x_left   = int( (x_center_bbox- width_bbox/2.) * width_image )
    x_right  = int( (x_center_bbox+ width_bbox/2.) * width_image )
    y_top    = int( (y_center_bbox-height_bbox/2.) * height_image )
    y_bottom = int( (y_center_bbox+height_bbox/2.) * height_image )

    return category, x_left, y_top, x_right, y_bottom

def parse_gt_detections(file_path):
    files = os.listdir(file_path)
    ground_truth = {}
    for filename in files:
        frame_id = int(filename.strip().split(".")[0])-1
        ground_truth[frame_id] = []
        content = open(os.path.join(file_path, filename), "r")
        obj_lines = content.readlines()
        for obj_line in obj_lines:
            ground_truth[frame_id].append(label2coord(obj_line.strip().split(), img_h, img_w))
        ground_truth[frame_id].sort(key=lambda x: x[1])
    return ground_truth

def parse_detections(file_path):
    detections = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:-1]:
            if line.strip() != "":
                line = line.strip().replace(" - ", "#").split("#")[1]
                if line.startswith('frame'):
                    frame_id = int(line.split(':')[0].split('frame')[1])
                    objects = eval(line.split(':')[1].strip())
                    detections[frame_id] = []
                    for obj in objects:
                        xywh = xywh2xyxy(obj[2])
                        detections[frame_id].append([obj[0], xywh, obj[3]])
                    detections[frame_id].sort(key=lambda x: x[1][0])
                    # detections[frame_id] = objects

    return detections



def calculate_metrics(ground_truth_, detection, iou_threshold=0.65):
    tp = 0
    fp = 0
    fn = 0
    frames = len(detection)
    Num_frame = frames
    Num_obj = len(ground_truth_[0])

    ground_truth = {k: ground_truth_[k] for k in range(0, frames)}
    
    for frame_id in ground_truth:
        gt_boxes = ground_truth[frame_id]
        det_boxes = detection.get(frame_id, [])

        matched_gt = [False] * len(gt_boxes)  # Track matched ground truth boxes
        matched_det = [False] * len(det_boxes)  # Track matched detection boxes

        for j, det in enumerate(det_boxes):
            det_box = det[1]  # Get the bounding box coordinates (ignore category)
            for i, gt in enumerate(gt_boxes):
                if not matched_gt[i]:  # Only consider unmatched ground truth boxes
                    gt_box = gt[1:]  # Get the bounding box coordinates (ignore category)
                    iou = calculate_iou(gt_box, det_box)
                    if iou >= iou_threshold:
                        tp += 1
                        matched_gt[i] = True  # Mark this ground truth box as matched
                        matched_det[j] = True  # Mark this detection box as matched
                        break

        fn += matched_gt.count(False)  # Count unmatched ground truth boxes
        fp += matched_det.count(False)  # Count unmatched detection boxes
        
    return tp, fp, fn, Num_frame, Num_obj

def calculate_id_switches(detection, iou_threshold=0.65):
    previous_frame_ids = {}
    id_switches = 0

    for frame_id in sorted(detection.keys()):
        current_frame_ids = {}
        current_detections = sorted(detection[frame_id], key=lambda x: x[2])  # Sort by object_id
        
        matched_ids = set()

        for det in current_detections:
            object_id = det[2]
            det_box = det[1]

            matched = False
            for prev_object_id, prev_box in previous_frame_ids.items():
                if prev_object_id in matched_ids:
                    continue  # Skip already matched IDs

                iou = calculate_iou(det_box, prev_box)
                if iou >= iou_threshold:
                    matched_ids.add(prev_object_id)  # Mark this ID as matched

                    if object_id != prev_object_id:
                        id_switches += 1
                    current_frame_ids[object_id] = det_box
                    matched = True
                    break

            if not matched:
                current_frame_ids[object_id] = det_box

        previous_frame_ids = current_frame_ids

    return id_switches


# Main function
if __name__ == "__main__":
    ground_truth_file = r"/content/TraceTalk/ByteTrack/datasets/zebrafish/ZebraFish-10/labelT"  # update with the correct path
    detection_file = r""/content/TraceTalk/log.txt"

    ground_truth = parse_gt_detections(ground_truth_file)
    detection = parse_detections(detection_file)
    
    metrics = calculate_metrics(ground_truth, detection, 0.3)
    IDs = calculate_id_switches(detection, 0.3)
    
    Precision = round(metrics[0] / (metrics[0] + metrics[1]), 4)
    Recall = round(metrics[0] / (metrics[0] + metrics[2]), 4)
    F1_score = round(2 * (Precision * Recall / (Precision + Recall)), 4)

    GT = metrics[3] * metrics[4]
    DIV_UP = metrics[1] + metrics[2] + IDs
    MOTA = 1 - DIV_UP / GT

    print(f"TP: {metrics[0]}")
    print(f"FP: {metrics[1]}")
    print(f"FN: {metrics[2]}")
    print(f"Precision: {Precision}")
    print(f"Recall: {Recall}")
    print(f"F1-score: {F1_score}")
    print("")
    print(f"ID Switch: {IDs}")
    print(f"MOTA: {MOTA}")

    
