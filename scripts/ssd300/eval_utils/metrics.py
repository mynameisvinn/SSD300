import numpy as np

def compute_recall_precision(evaluator, n_classes):
	pos = evaluator.num_gt_per_class
    total_tp = []
    total_fp = []
    tp = []
    fp = []
    class_recall = []
    class_precision = []
    for th in np.arange(0, 1, 0.01):
        class_tp = []
        class_fp = []
        recall = []
        precision = []
        result_trio = zip(evaluator.true_positives,
                          evaluator.false_positives,
                          evaluator.prediction_results)
        for i, (etp, efp, pred_results) in enumerate(result_trio):
            if i != 0: # Skip the background class
                if len(pred_results) != 0:
                    class_tp.append(sum(etp[np.array(pred_results)[:, 1] > th]))
                    class_fp.append(sum(efp[np.array(pred_results)[:, 1] > th]))
                    recall.append(1.0 * class_tp[i - 1] / pos[i])
                    if (class_tp[i-1] + class_fp[i-1]) == 0:
                        precision.append(0)
                    else:
                        precision.append(1.0 * class_tp[i-1] / (class_tp[i-1] + class_fp[i-1]))
                else:
                    class_tp.append(0)
                    class_fp.append(0)
                    recall.append(0)
                    precision.append(0)
        class_recall.append(recall)
        class_precision.append(precision)
        tp.append(class_tp)
        fp.append(class_fp)
    class_recall = np.array(class_recall)
    class_precision = np.array(class_precision)
    tp = np.array(tp)
    fp = np.array(fp)
    total_tp = tp.sum(1)
    total_fp = fp.sum(1)
    recall = total_tp/sum(pos)
    precision = np.where((total_tp + total_fp) == 0, 0, total_tp/(total_tp + total_fp))
    return {'positives': pos
    		'class_recall': class_recall,
    		'class_precision': class_precision,
    		'tp': tp,
    		'fp': fp}