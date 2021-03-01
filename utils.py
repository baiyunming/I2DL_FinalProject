import torch
from collections import Counter

def iou(box_predict, box_target):
    """
    // yolo version prediction and target box
    input:
    box_predict -- predicted box, dim: (batch_size, S, S, 4) [midpoint_x. midpoint_y, width, height]
    box_target -- target box, dim: (batch_size, S, S, 4) [midpoint_x, midpoint_y, width, height]

    output: intersection of union
    """
    #find corner point of intersection region
    box1_x1 = box_predict[...,0:1] - 0.5 * box_predict[...,2:3] #preserve last dimension shape(batch_size, S, S, 1)
    box1_y1 = box_predict[...,1:2] - 0.5 * box_predict[...,3:4]
    box1_x2 = box_predict[...,0:1] + 0.5 * box_predict[...,2:3]
    box1_y2 = box_predict[...,1:2] + 0.5 * box_predict[...,3:4]

    box2_x1 = box_target[...,0:1] - 0.5 * box_target[...,2:3]
    box2_y1 = box_target[...,1:2] - 0.5 * box_target[...,3:4]
    box2_x2 = box_target[...,0:1] + 0.5 * box_target[...,2:3]
    box2_y2 = box_target[...,1:2] + 0.5 * box_target[...,3:4]

    xi1 = torch.max(box1_x1, box2_x1)
    yi1 = torch.max(box1_y1, box2_y1)
    xi2 = torch.min(box1_x2, box2_x2)
    yi2 = torch.min(box1_y2, box2_y2)

    inter_area = (yi2 - yi1).clamp(0)* (xi2 - xi1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou

def non_max_suppression(pred_boxes, iou_threshold_nms, probability_threshold):
    """"
    # non_max_suppression only operate on predicted boxes
    imput:
    pred_boxes:list of element [class_idx, object_probability, x, y, w, h]
    iou_threshold: bounding_box predicting same class, whose iou with box with largest object_probability larger than this threshold will be discarded
    probability_threshold: bounding_box with object_probability les than this threshold will be discarded

    output: filtered list of bounding boxes
    """
    assert type(pred_boxes) == list

    #discard bounding boxes with low object_probability
    bboxes = [box for box in pred_boxes if box[1] > probability_threshold]
    bboxes.sort(key=take_Object_Probability, reverse=True)
    bboxes_after_nms = []

    while bboxes:
        first_box = bboxes.pop(0)
        #remain bounding_boxes with different class or same class with less IOU
        bboxes = [box for box in bboxes if box[0] != first_box[0] or iou(torch.tensor(first_box[2:]),torch.tensor(box[2:]))<iou_threshold_nms]
        bboxes_after_nms.append(first_box)

    return bboxes_after_nms

# take second element for sort
def take_Object_Probability(elem):
    return elem[1]


def mean_average_precision (pred_boxes, true_boxes, iou_threshold, num_classes=4):
    #pred_boxes after non_max_suppression
    #list of all bounding_boxes [[train_idx, class_prob, prob_score, x, y, w, h], [], [] ...]
    average_precision = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        #all predicted boxes for specific class
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        #all ground_truth boxes for specific class
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        #amount_bboxes: dictionary
        #key: idx of predicted image
        #value: zeros tensor with size number of gt_boxes
        #indicate whether the gt_box is already associated with one predicted box (1: yes, 0: no)
        #ensure one target_box correspond to only one prediction
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, value in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(value)

        #sort all detections according to its confidence score
        detections.sort(key=lambda x:x[2], reverse = True)
        #used to calculate precision
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths) #used for calculating recall

        for detection_idx, detection in enumerate(detections):
            #find same all gt_bounding with the same frame
            gt_bboxes_currentframe = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            #num_gt_bboxes_currentframe = len(gt_bboxes_currentframe)

            best_iou = 0
            best_gt_idx = -1

            #calculate iou between current detection with all gt_boxes in current_frame
            for idx, gt in enumerate(gt_bboxes_currentframe):
                iou_result = iou(torch.tensor(detection[3:]), torch.tensor(gt[3:]))
                if iou_result > best_iou:
                    best_iou = iou_result
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else :
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        #precision, recall calculation
        TP_cumsum = torch.cumsum(TP, dim = 0)
        FP_cumsum = torch.cumsum(FP, dim = 0)

        recall = torch.div(TP_cumsum, (total_true_bboxes + epsilon))
        precision = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        precision = torch.cat((torch.tensor([1]), precision))
        recall = torch.cat((torch.tensor([0]), recall))

        average_precision.append(torch.trapz(precision, recall))


    return sum(average_precision)/len(average_precision)


def get_list_boxes(prediction, S=5):
  #prediction dim: batch_size * S * S * 14 （class_idx 4 + 5 * B=2）
  #return list: batch_size * (S*S) 
  batch_size = prediction.shape[0]
  prediction = prediction.to("cpu")
  prediction = prediction.reshape(prediction.shape[0], S, S, 14)
  pred_box1 = prediction[...,5:9]
  pred_box2 = prediction[...,10:14]

  cat_result = torch.cat([(prediction[...,4]).unsqueeze(0), prediction[...,9].unsqueeze(0)],dim=0)
  box_high_score = cat_result.argmax(0).unsqueeze(-1)
  final_box =(1-box_high_score) * pred_box1 + box_high_score * pred_box2

  idx_x = torch.arange(S).repeat(batch_size,S,1)
  idx_y = torch.arange(S).unsqueeze(-1).repeat(batch_size,1,S)

  # dim: batch_size * S * S * 1 
  x_wrt_image = (1/S) * (final_box[...,0:1] + idx_x.unsqueeze(-1)) 
  y_wrt_image = (1/S) * (final_box[...,1:2] + idx_y.unsqueeze(-1)) 
  width_wrt_image = (1/S) * final_box[...,2:3]
  height_wrt_image = (1/S) * final_box[...,3:4]
  clss_label = prediction[...,0:4].argmax(-1).unsqueeze(-1) #get class label
  confidence = torch.max(prediction[...,4], prediction[...,9]).unsqueeze(-1)

  converted_box = torch.cat([clss_label, confidence, x_wrt_image, y_wrt_image, width_wrt_image, height_wrt_image],dim=-1)

  reshaped_box = converted_box.reshape(batch_size, S*S, -1) #batch_size, S*S, 6 [class_idx, object_probability, x, y, w, h]
  #print(reshaped_box.shape)

  boxes_batch = []
  for i in range(batch_size):
    boxes_frame = []
    for j in range(S*S):
      boxes_frame.append([x.item() for x in reshaped_box[i,j,:]])
    boxes_batch.append(boxes_frame)

  
  return boxes_batch 


def calculate_map(model, train_dataloader, iou_threshold, confidence_threshold):
  model.eval()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  all_predict_boxes = []
  all_true_boxes = []
  frame_idx = 0

  for batch_idx, (image, labels, _) in enumerate(train_dataloader):
    with torch.no_grad():
      prediction = model(image.to(device))
  
    predict_boxes_batch = get_list_boxes(prediction, S=5)
    true_boxes_batch = get_list_boxes(labels.to(device), S=5)
    batch_size = prediction.shape[0]

    for i in range(batch_size):
      predict_boxes_frame = predict_boxes_batch[i]

      #nms_predict_boxes_frame = non_max_suppression(predict_boxes_frame,iou_threshold,confidence_threshold)
      nms_predict_boxes_frame = non_max_suppression(predict_boxes_frame,0.3,confidence_threshold)

      for j in range(len(nms_predict_boxes_frame)):
        all_predict_boxes.append([frame_idx] + nms_predict_boxes_frame[j])

      true_boxes_frame = true_boxes_batch[i]
      for k in range(len(true_boxes_frame)):
        if true_boxes_frame[k][1] > confidence_threshold:
          all_true_boxes.append([frame_idx] + true_boxes_frame[k])
      frame_idx = frame_idx + 1

  map = mean_average_precision(all_predict_boxes, all_true_boxes, iou_threshold)
  return map
  


