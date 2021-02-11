import os
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger, LearningRateScheduler

from ..keras_loss_function.keras_ssd_loss import SSDLoss
from .keras_ssd300 import ssd_300
from ..ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ..keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ..keras_layers.keras_layer_L2Normalization import L2Normalization

from ..data_generator.object_detection_2d_geometric_ops import Resize
from ..data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from ..data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

import neptune

import logging
LOG = logging.getLogger(__name__)

# 1: Build the Keras model.
img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [170, 170, 170] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = [0.01, 0.16, 0.32, 0.64, 1.1] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = True # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords=True
K.clear_session() # Clear previous models from memory.


def create_ssd300(n_classes, mode, start_lr=0.001):
    if mode == 'training':
        model = ssd_300(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        mode='training',
                        l2_regularization=0.0005,
                        scales=scales_coco,
                        aspect_ratios_per_layer=aspect_ratios,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=steps,
                        offsets=offsets,
                        clip_boxes=clip_boxes,
                        variances=variances,
                        normalize_coords=normalize_coords,
                        subtract_mean=mean_color,
                        swap_channels=swap_channels)
    # 3: Instantiate an optimizer and the SSD loss function and compile the model.
        adam = Adam(lr=start_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        return model
    elif mode == 'evaluation':
        model = ssd_300(image_size=(img_height, img_width, 3),
                        n_classes=n_classes,
                        mode='inference_fast',
                        l2_regularization=0,
                        scales=scales_pascal,
                        aspect_ratios_per_layer=aspect_ratios,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=steps,
                        offsets=offsets,
                        clip_boxes=clip_boxes,
                        variances=variances,
                        normalize_coords=normalize_coords,
                        subtract_mean=mean_color,
                        swap_channels=swap_channels,
                        confidence_thresh=0.5,
                        iou_threshold=0.5,
                        top_k=200,
                        nms_max_output_size=400)
        return model
    else:
        raise Exception('Invalid mode. Choose either evaluation or training')


def reload_ssd300(model_path):
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    return load_model(model_path, custom_objects={
        'AnchorBoxes': AnchorBoxes,
        'L2Normalization': L2Normalization,
        'compute_loss': ssd_loss.compute_loss})


def create_ssd_encoder(model, n_classes):
    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                       model.get_layer('fc7_mbox_conf').output_shape[1:3],
                       model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]
    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales_coco,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=normalize_coords)
    return ssd_input_encoder


def data_aug(params=None):
    da = SSDDataAugmentation(
        img_height=img_height,
        img_width=img_width,
        background=mean_color)
    if params is not None:
        da.sequence[0].random_brightness.prob = params['random_brightness']
        da.sequence[0].random_channel_swap.prob = params['random_channel_swap']
        da.sequence[0].random_contrast.prob = params['random_contrast']
        da.sequence[0].random_hue.prob = params['random_hue']
        da.sequence[0].random_saturation.prob = params['random_saturation']
        da.sequence[1].blur_prob = params['random_blur']
        da.sequence[1].dropout_prob = params['random_dropout']
        da.sequence[2].expand.prob = params['random_expand']
        da.sequence[3].random_crop.prob = params['random_crop']
        da.sequence[4].prob = params['random_flip']
        da.sequence[5].prob = params['random_rotate']
    return da

def resize():
    return Resize(height=img_height, width=img_width)

def convert_to_3_channels():
    return ConvertTo3Channels()


def create_lr_schedule(start_lr=0.001, lr_decay=0.001, lr_rate=100, min_lr=0.000001):
    def lr_schedule(epoch):
        new_lr = start_lr * np.power(lr_decay,int(epoch / lr_rate)) 
        if new_lr < min_lr:
           return min_lr
        return new_lr
    return lr_schedule


class Neptune_Callback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        neptune.log_metric(log_name='learning_rate', x=epoch, y=float(K.eval(self.model.optimizer.lr)))
        neptune.log_metric(log_name=f'train_loss', x=epoch, y=logs['loss'])
        neptune.log_metric(log_name=f'validation_loss', x=epoch, y=logs['val_loss'])


def train_callbacks(output_path, auto_lr, start_lr, lr_decay, lr_rate, min_lr, iters, neptune_callback=False):
    callbacks = []
    callbacks.append(ModelCheckpoint(
        filepath=os.path.join(
        output_path,
        'checkpoint_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1))
    callbacks.append(CSVLogger(filename=os.path.join(output_path, f'ssd300_training_log.csv'),
                               separator=',',
                               append=True))
    if auto_lr:
        learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss',
                                                 factor=lr_decay,
                                                 patience=lr_rate,
                                                 verbose=1,
                                                 epsilon=0.001,
                                                 cooldown=0,
                                                 min_lr=min_lr)
    else:
        learning_rate_scheduler = LearningRateScheduler(schedule=create_lr_schedule(start_lr, lr_decay, lr_rate, min_lr))
    callbacks.append(learning_rate_scheduler)
    callbacks.append(TerminateOnNaN())
    if neptune_callback:
        callbacks.append(Neptune_Callback())
    return callbacks


def load_best_checkpoint(model, checkpoint_folder):
    best_val_loss = np.inf
    best_checkpoint = ''
    for checkpoint_path in os.listdir(checkpoint_folder):
        val_loss = float(checkpoint_path.split('-')[-1][:-3])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = checkpoint_path
    model.load_weights(
        os.path.join(checkpoint_folder, best_checkpoint), by_name=True)