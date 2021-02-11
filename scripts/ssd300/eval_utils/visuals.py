import os
import numpy as np
from matplotlib import pyplot as plt, cm, patches

from . import eval_tools

import timeit
import logging
LOG = logging.getLogger(__name__)

import neptune


def plot_bounding_boxes(images, labels, label_index, style='ssd', titles=None, save_path=None,
                        label_format={'class_id': 0, 'score': 1, 'x1': 2, 'y1': 3, 'x2': 4, 'y2': 5}):
    nrows = int(len(images) / 2) + 1
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        plot_image(image=images[i], label=labels[i],
            label_index=label_index, style=style, title=titles[i],
            fig=fig, ax=ax, 
            save_path=None, label_format=label_format)
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_image(image, label=None, label_index=None, style='ssd', title='plot', fig=None, ax=None, 
               save_path=None, label_format={'class_id': 0, 'x1': 1, 'y1': 2, 'x2': 3, 'y2': 4}):
    """Convenient plotting function"""
    if fig is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()
    ax.imshow(image, cmap=cm.get_cmap('Greys', 256).reversed())
    ax.set_title(title)
    if label is not None:
        if label_index is not None:
            colors = plt.cm.hsv(np.linspace(0, 1, len(label_index)+1)).tolist()
        else:
            class_ind = label_format['class_id']
            colors  = {l[class_ind]: 'r' for l in label}
        for lab in label:
            l = int(lab[label_format['class_id']])
            if label_index is not None:
                label_text = label_index[l]
            else:
                label_text = l
            if 'score' in label_format:
                score = lab[label_format['score']]
                label_text = '{}: {:.2f}%'.format(label_text, score * 100)
            # Bounding Box
            x1 = lab[label_format['x1']]
            y1 = lab[label_format['y1']]
            x2 = lab[label_format['x2']]
            y2 = lab[label_format['y2']]
            if style == 'yolo':
                rect = patches.Rectangle((x1,y1), x2, y2, linewidth=2, edgecolor=colors[l], facecolor='none')
            elif style == 'ssd':
                rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor=colors[l], facecolor='none')
            else:
                raise Exception('Invalid style for bounding box')
            ax.add_patch(rect)
            ax.text(x1, y1, label_text, color='black', bbox={'facecolor': colors[l], 'alpha': 0.5})
    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_average_precision(results, label_index, save_path=None, neptune_logger=None):
    MAP = results[0]
    average_precision = results[1]
    n_classes = len(label_index)
    fig = plt.figure(figsize=(10,10))
    plt.bar(np.arange(1, n_classes + 1), average_precision[1:(n_classes + 1)])
    plt.xlabel('class', color='black', fontsize=20)
    plt.ylabel('precision', color='black', fontsize=20)
    plt.xticks(np.arange(n_classes),
               labels=[label_index[i] for i in np.arange(1, n_classes + 1)],
               rotation=45,
               color='black',
               fontsize=15)
    plt.ylim((0,1))
    plt.title(f'Average Precision SSD300 (MAP: {round(MAP, 4)})', color='black', fontsize=20)

    if save_path:
        fig.savefig(os.path.join(save_path, 'average_precision.png'))
    if neptune_logger is not None:
        neptune_logger.log_metric('MAP', round(MAP, 4))
        neptune_logger.log_image('Average Precision', x=0, y=fig, image_name='Average Precision per class')


def plot_RC_curve(stats, label_index, save_path=None, neptune_logger=None):
    # Compute Recall, Precision, F1 for each threshold value
    fig = plt.figure(figsize=(8,8))
    plt.plot(np.arange(0,1,0.01), stats['recall'], color='blue')
    plt.plot(np.arange(0,1,0.01), stats['precision'], color='orange')
    plt.plot(np.arange(0,1,0.01), stats['f1'], color='red')
    plt.title(f'Recall Precision Curve', color='black', fontsize=20)
    plt.xlabel(f'Threshold', color='black', fontsize=20)
    plt.ylabel(f'Probability', color='black', fontsize=20)
    plt.xticks(color='black', fontsize=20)
    plt.yticks(color='black', fontsize=20)
    recall = stats['recall'][0]
    precision = stats['precision'][0]
    f1 = stats['f1'][0]
    recall_text = '{:.2f}%'.format(recall*100)
    precision_text = '{:.2f}%'.format(precision*100)
    f1_text = '{:.2f}%'.format(f1*100)
    plt.text(0, recall, recall_text, fontsize=14, color='blue', horizontalalignment='left', verticalalignment='top')
    plt.text(0.1, precision, precision_text, fontsize=14, color='orange', horizontalalignment='center', verticalalignment='bottom')
    plt.text(0.2, f1, f1_text, fontsize=12, color='red', horizontalalignment='right', verticalalignment='center')
    plt.ylim((0,1))
    plt.legend(['Recall', 'Precision', 'F1'])
    if save_path:
        fig.savefig(os.path.join(save_path, 'recall_precision_curve.png'))

    # Plot Recall Precision curve per class
    n_classes = len(label_index) #number of non-background classes
    ncols = 4
    nrows = int(n_classes / ncols) + 1
    fig2, ax2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    ax2 = ax2.flatten()
    for i in range(1, n_classes + 1): # Ignore background class
        class_recall = stats['class_recall'][:, i]
        class_precision = stats['class_precision'][:, i]
        class_f1 = stats['class_f1'][:, i] 
        ax2[i].plot(class_recall, color='blue')
        ax2[i].plot(class_precision, color='orange')
        ax2[i].plot(class_f1, color='red')        
        recall_text = '{:.2f}%'.format(class_recall[0]*100)
        precision_text = '{:.2f}%'.format(class_precision[0]*100)
        f1_text = '{:.2f}%'.format(class_f1[0]*100)
        ax2[i].text(10, class_recall[0], recall_text, fontsize=10, color='blue', horizontalalignment='center', verticalalignment='bottom')
        ax2[i].text(20, class_precision[0], precision_text, fontsize=10, color='orange', horizontalalignment='center', verticalalignment='top')
        ax2[i].text(30, class_f1[0], f1_text, fontsize=10, color='red', horizontalalignment='center', verticalalignment='center')
        ax2[i].set_title(f' {label_index[i]} Recall-Precision Curve', color='black', fontsize=12)
        ax2[i].tick_params(axis='y', colors='black', labelsize=12)
        ax2[i].tick_params(axis='x', colors='black')
        ax2[i].set_ylim((0,1))
    if save_path:
        fig2.savefig(os.path.join(save_path, 'recall_precision_class_curve.png'))
    if neptune_logger is not None:
        neptune_logger.log_metric('Recall', round(stats['recall'][0],4))
        neptune_logger.log_metric('Precision', round(stats['precision'][0],4))
        neptune_logger.log_metric('F1-Score', round(stats['f1'][0],4))
        for i in range(1, n_classes + 1):
            neptune_logger.log_metric(f'{label_index[i]} Recall', round(stats['class_recall'][0, i], 4))
            neptune_logger.log_metric(f'{label_index[i]} Precision', round(stats['class_precision'][0, i], 4))
        neptune_logger.log_image('Recall-Precision', x=fig, image_name='Recall-Precision Curve')
        neptune_logger.log_image('Recall-Precision Per Class', x=fig2, image_name='Recall-Precision Curve Per Class')
    return fig, fig2, ax2


def plot_sample_results(image_labels, test_x, test_y, ind_id_to_plot, label_index, thresh=0.0, save_path=None, neptune_logger=None):
    # 5: Draw the predicted boxes onto the image
    grey_cmap = cm.get_cmap('Greys', 256).reversed()
    ncols = 2
    nrows = len(ind_id_to_plot)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols, 5*nrows))
    if nrows == 1:
        ax = ax.reshape(1,2)
    colors = plt.cm.hsv(np.linspace(0, 1, len(label_index)+1)).tolist() # Set the colors for the bounding boxes
    for i, t in enumerate(ind_id_to_plot):
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        ax[i, 0].imshow(np.array(test_x[t[0]]), cmap=grey_cmap)
        ax[i, 0].set_title(f'Image: {t} ground truth', color='black')
        ax[i, 1].imshow(np.array(test_x[t[0]]), cmap=grey_cmap)
        ax[i, 1].set_title(f'Image: {t} model predicted', color='black')
        for box in test_y[t[0]]:
            label = label_index[int(box[0])]
            xmin = box[1]
            ymin = box[2]
            xmax = box[3]
            ymax = box[4]
            color = colors[int(box[0])]
            ax[i, 0].add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
            ax[i, 0].text(xmin, ymin, label, size='small', color='black', bbox={'facecolor':color, 'alpha':0.5})
        for pred_box in image_labels[t[1]]:
            label = label_index[int(pred_box[0])]
            conf = pred_box[1]
            if conf > thresh:
                xmin = pred_box[2]
                ymin = pred_box[3]
                xmax = pred_box[4]
                ymax = pred_box[5]
                color = colors[int(pred_box[0])]
                label = '{}: {:.2f} '.format(label, pred_box[1]*100)
                ax[i, 1].add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
                ax[i, 1].text(xmin, ymin, label, size='small', color='black', bbox={'facecolor':color, 'alpha':0.5})
    if save_path:
        fig.savefig(save_path)
    if neptune_logger is not None:
        neptune_logger.log_image(save_path,  x=fig, image_name=save_path)
    return fig, ax