import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import config

class DiceLoss(nn.Module):
    """
    Calculate Dice loss for each class separately and return the normalized average.
    
    This implementation supports class weighting to handle imbalanced datasets
    and can optionally ignore the background class. The loss is normalized to
    always be in the range [0, 1].

    Args:
        pred (torch.Tensor): Predictions from the model, torch.Size([1, 5, 150, 180, 155]) with logits for 5 classes at dim 1
        target (torch.Tensor): Ground truth labels, torch.Size([1, 150, 180, 155]) with class indices [0,1,2,3,4] at dim 0

        ignore_background (bool): Whether to ignore the background class (class 0), default is False

    Returns:
        torch.Tensor: Normalized weighted average Dice loss, a scalar value in the range [0, 1]
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), -1)

        start_class = 1 # set to 0 to include background
        
        dice_scores = []
        
        for class_idx in range(start_class, pred.size(1)):
            pred_class = pred[:, class_idx, ...]
            target_class = (target == class_idx).float()
            
            intersection = (pred_class * target_class).sum() # logical AND operation
            union = pred_class.sum() + target_class.sum()
            
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.append(dice)
        
        return 1 - torch.mean(torch.stack(dice_scores))

def calculate_metrics(pred, target, smooth=1e-5):
    """
    Calculate precision, recall, f1 score and dice coefficient for multi-class segmentation
    Compute metrics for each class separately and return the average across all classes
    Args:
    pred (torch.Tensor), torch.Size([1, 150, 180, 116]): With class indices [0,1,2,3,4] of maximum logits
    target (torch.Tensor), torch.Size([1, 150, 180, 116]): With class indices [0,1,2,3,4]
    smooth (float): Smoothing factor to avoid division by zero
    Returns:
    avg_precision (torch.Tensor): Average precision across all classes
    avg_recall (torch.Tensor): Average recall across all classes
    avg_f1 (torch.Tensor): Average F1 score across all classes
    avg_dice (torch.Tensor): Average Dice coefficient across all classes
    """
    num_classes = config.out_channels
    precision = torch.zeros(num_classes, device=pred.device)
    recall = torch.zeros(num_classes, device=pred.device)
    f1 = torch.zeros(num_classes, device=pred.device)
    dice = torch.zeros(num_classes, device=pred.device)
    
    for class_idx in range(1, num_classes): # Always ignore background class
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        intersection = (pred_class * target_class).sum() # logical AND operation
        pred_sum = pred_class.sum()
        target_sum = target_class.sum()
        
        precision[class_idx] = intersection / (pred_sum + smooth)
        recall[class_idx] = intersection / (target_sum + smooth)
        f1[class_idx] = 2 * precision[class_idx] * recall[class_idx] / (precision[class_idx] + recall[class_idx] + smooth)
        dice[class_idx] = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    avg_precision = torch.mean(precision)
    avg_recall = torch.mean(recall)
    avg_f1 = torch.mean(f1)
    avg_dice = torch.mean(dice)
    
    return avg_precision, avg_recall, avg_f1, avg_dice


def calculate_metrics_EVAL(pred, target):
    """
    Calculate precision, recall, f1 score and dice coefficient for multi-class segmentation
    Compute metrics for each class separately and return separately + the average

    CAVE: BACKGROUND class is excluded from the average metrics

    Args:
        pred (torch.Tensor), torch.Size([1, 150, 180, 116]): With class indices [0,1,2,3] of maximum logits at dim 0
        target (torch.Tensor), torch.Size([1, 150, 180, 116]): Wit class incides [0,1,2,3] at dim 0

    Returns:
        metrics (dict): Dictionary containing the metrics for each class and the average metrics
    """
    # Initialize metrics
    precision = torch.zeros(4)
    recall = torch.zeros(4)
    f1 = torch.zeros(4)
    dice = torch.zeros(4)

    # Calculate metrics for each class
    for class_idx in range(4):
        true_positive = torch.sum((pred == class_idx) & (target == class_idx))
        false_positive = torch.sum((pred == class_idx) & (target != class_idx))
        false_negative = torch.sum((pred != class_idx) & (target == class_idx))

        precision[class_idx] = true_positive / (true_positive + false_positive + 1e-7)
        recall[class_idx] = true_positive / (true_positive + false_negative + 1e-7)
        f1[class_idx] = 2 * precision[class_idx] * recall[class_idx] / (precision[class_idx] + recall[class_idx] + 1e-7)
        dice[class_idx] = 2 * true_positive / (2 * true_positive + false_positive + false_negative + 1e-7)

    # Calculate average metrics
    avg_precision = torch.mean(precision[1:])
    avg_recall = torch.mean(recall[1:])
    avg_f1 = torch.mean(f1[1:])
    avg_dice = torch.mean(dice[1:])

    # Create dictionary of metrics
    metrics = {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'avg_dice': avg_dice,
        'precision_background': precision[0],
        'precision_outer_tumour': precision[1],
        'precision_enhancing_tumour': precision[2],
        'precision_tumour_core': precision[3],
        'recall_background': recall[0],
        'recall_outer_tumour': recall[1],
        'recall_enhancing_tumour': recall[2],
        'recall_tumour_core': recall[3],
        'f1_background': f1[0],
        'f1_outer_tumour': f1[1],
        'f1_enhancing_tumour': f1[2],
        'f1_tumour_core': f1[3],
        'dice_background': dice[0],
        'dice_outer_tumour': dice[1],
        'dice_enhancing_tumour': dice[2],
        'dice_tumour_core': dice[3]
    }

    return metrics