import numpy as np
from .average_precision_evaluator import Evaluator

img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images

def get_evaluate_func(model, n_classes, label_convert=None, apply_unique_box=False, apply_adjacent_rule=False):

    def evaluate(generator, num_to_test):
        evaluator = Evaluator(
            model=model, n_classes=n_classes, data_generator=generator,
            model_mode='inference', apply_unique_box=apply_unique_box, apply_adjacent_rule=apply_adjacent_rule)
        results = evaluator(
            img_height=img_height, img_width=img_width,
            batch_size=num_to_test,
            label_convert=label_convert,
            data_generator_mode='resize',
            round_confidences=2,
            matching_iou_threshold=0.5,
            border_pixels='include',
            sorting_algorithm='quicksort',
            average_precision_mode='integrate',
            num_recall_points=256,
            ignore_neutral_boxes=True,
            return_precisions=True,
            return_recalls=True,
            return_average_precisions=True,
            verbose=True)
        return evaluator, results
    return evaluate


def compute_recall_precision_curves(class_num_gt, true_positives, false_positives, prediction_results):
    total_tp, total_fp, tp, fp, class_recall, class_precision = [], [], [], [], [], []
    for th in np.arange(0, 1, 0.01):
        class_tp, class_fp, recall_th, precision_th = [], [], [], []
        result_trio = zip(true_positives,
                          false_positives,
                          prediction_results)
        for i, (etp, efp, pred_results) in enumerate(result_trio):
            if len(pred_results) != 0:
                scores = np.array(pred_results)[:, 1].astype(float)
                class_tp.append(sum(etp[scores > th]))
                class_fp.append(sum(efp[scores > th]))
                recall_th.append(1.0 * class_tp[i] / class_num_gt[i])
                sum_positives = class_tp[i] + class_fp[i]
                if (sum_positives) == 0:
                    precision_th.append(0)
                else:
                    precision_th.append(1.0 * class_tp[i] / (sum_positives))
            else:
                class_tp.append(0)
                class_fp.append(0)
                recall_th.append(0)
                precision_th.append(0)
        class_recall.append(recall_th)
        class_precision.append(precision_th)
        tp.append(class_tp)
        fp.append(class_fp)
    class_recall = np.array(class_recall)
    class_precision = np.array(class_precision)
    total_class_recall_precision = class_recall + class_precision
    class_f1 = np.where(total_class_recall_precision == 0 ,
        0, 2 * (class_recall * class_precision) / (total_class_recall_precision))
    tp = np.array(tp)
    fp = np.array(fp)
    total_tp = tp.sum(1)
    total_fp = fp.sum(1)
    total_detections = total_tp + total_fp
    recall = total_tp/sum(class_num_gt)
    precision = np.where((total_detections) == 0,
        0, total_tp/(total_detections))
    total_recall_precision = recall + precision
    f1 = np.where(total_recall_precision == 0,
        0, 2*(recall * precision)/(total_recall_precision))
    return {'class_recall': class_recall,
            'class_precision': class_precision,
            'class_f1': class_f1,
            'true_positives': tp,
            'false_positives': fp,
            'recall': recall,
            'precision': precision,
            'f1': f1}


def process_results_per_image(results, class_ind_focus=None):
    image_labels = {}
    for class_ind, class_label in enumerate(results):
        if ((class_ind_focus is not None) and (class_ind not in class_ind_focus)) or len(class_label) == 0:
            continue
        for image_label in class_label:
            image_id = image_label[0]
            label_box = [class_ind] + list(image_label[1:])
            if image_id not in image_labels:
                image_labels[image_id] = [label_box]
            else:
                image_labels[image_id].append(label_box)
    return image_labels


def identify_missed_detections(image_labels, test_image_ids, test_y, class_ind_focus=None, false_negatives=None):
    miss_index = []
    for ind, image_id in enumerate(test_image_ids):
        if image_id not in image_labels:
            continue
        predicted_classes = np.array(image_labels[image_id])[:, 0]
        ground_truth_classes = np.array(test_y[ind])[:, 0]
        if class_ind_focus is not None and false_negatives is not None:
            if class_ind_focus in ground_truth_classes and image_id not in false_negatives[class_ind_focus]:
                miss_index((ind, image_id))
                continue
        ground_truth_classes.sort()
        predicted_classes.sort()
        if ground_truth_classes.shape != predicted_classes.shape or (ground_truth_classes != predicted_classes).any():
            miss_index.append((ind, image_id))
    return miss_index