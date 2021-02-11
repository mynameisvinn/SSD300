import os
import numpy as np
import skimage.transform
from matplotlib import pyplot as plt, cm, colors
from matplotlib import animation
import tensorflow.keras as keras

# Create colormaps for original frame

GREY_CM = cm.get_cmap('Greys', 256).reversed()
# Create colormaps for cam
jet_cm = cm.get_cmap('jet', 256)
jet_colors = jet_cm(np.linspace(0, 1, 256))
# Increase alpha as value gets larger
jet_colors[:, 3] = np.power(np.linspace(0, 0.6, 256), 1)
CAM_CM = colors.ListedColormap(jet_colors)


def class_activation_maps(frames,
                          model,
                          label_dict,
                          label_to_plot=None,
                          num_frames=None,
                          penult_layer_name='penult_conv_layer'):

    """Create Class Activation map matrices
    Args:

        frames: NumPy Array (N, M, N, C)

        model: Keras model object

        schema: dict of all level labels

        label: dict with frame-level labels to compute cams

        num_frames: number of frames to compute. If None all are computed

        penult_layer_name: name of last conv layer in model

 

    Return:

        dict of cams with levels and classes as the key

    """

    frames = np.array(frames)
    frame_size = frames.shape[1:3]
    nf = frames.shape[0]
    if num_frames is None:
        frame_ind = np.arange(nf)
    else:
        interval = int(nf / num_frames) if nf >= num_frames else 1
        frame_ind = np.arange(0, nf, interval)
    frames = frames[frame_ind]
    # Get layer activations
    activations = get_activations(
        frames, model, penult_layer_name=penult_layer_name)
    num_filters = activations.shape[3]
    # Get softmax weights
    softmax_weights = get_softmax_weights(model)
    # Color frame in Grey colormap
    frames_grey = np.array(resize_and_color_many_frames(
        frames=frames[:, :, :, 0], colormap=GREY_CM, frame_size=frame_size))
    level_cams = {}
    # if label_to_plot is None:  # color all the cams
    class_cams = []
    # Compute cam for each class
    classes = list(label_dict.keys())
    for class_name in classes:
        class_weights = softmax_weights[:num_filters, label_dict[class_name]]
        class_cams.append(activations * class_weights)
    class_cams = np.array(class_cams).sum(-1)
    # Obtain global max and min value
    cam_max = class_cams.max()
    cam_min = class_cams.min()
    cam_color = [resize_and_color_many_frames(
        frames=cams,
        colormap=CAM_CM,
        frame_size=frame_size,
        norm_max=cam_max,
        norm_min=cam_min)
        for cams in class_cams]
    # Blend grey image with cam
    level_cams[level] = {
        class_label: blend_many_frames(frames_grey, frames_cam)
        for class_label, frames_cam in zip(classes, cam_color)}
    '''
    else:
        class_cams = []
        for fi in range(frames.shape[0]):
            class_ind = level_schema.index(label[level][fi])
            class_weights = softmax_weights[level][:num_filters, class_ind]
            cam_calc = activations[fi] * class_weights
            class_cams.append(cam_calc)
        class_cams = np.array(class_cams).sum(-1)
        # Obtain global max and min value
        cam_max = class_cams.max()
        cam_min = class_cams.min()
        level_cams[level] = color_and_blend_frames(
            cam_frames=class_cams,
            input_frames=frames_grey,
            colormap=CAM_CM,
            cam_max=cam_max,
            cam_min=cam_min)
    '''
    return level_cams
 

def get_activations(frames,
                    model,
                    penult_layer_name='activation_238'):
    inp = model.input
    penult_conv_layer = model.get_layer(penult_layer_name)
    output = penult_conv_layer.output
    activation_function = keras.backend.function(inp, output)
    activations = activation_function(frames)
    return activations
 

def get_softmax_weights(model):
    level_softmax_layer = model.get_layer('softmax')
    level_weights = level_softmax_layer.get_weights()[0]
    return level_weights
 

def color_and_blend_frames(cam_frames,
                           input_frames,
                           colormap,
                           cam_max,
                           cam_min):

    cb_frames = []
    frame_size = input_frames[0].shape[:2]
    for cam_f, inp_f in zip(cam_frames, input_frames):
        rc_frame = resize_and_color_frame(
            cam_f, colormap, frame_size, cam_max, cam_min)
        blended_frames = blend_frames(inp_f, rc_frame)
        cb_frames.append(blended_frames)
    return cb_frames
 

def resize_and_color_many_frames(frames,
                                 colormap,
                                 frame_size=None,
                                 norm_max=None,
                                 norm_min=None):
    rc_frames = [resize_and_color_frame(
        ff, colormap, frame_size, norm_max, norm_min) for ff in frames]
    return rc_frames
 

def resize_and_color_frame(frame,
                           colormap,
                           frame_size=None,
                           norm_max=None,
                           norm_min=None):
    resized_normalized_frame = norm_max_min(skimage.transform.resize(
        frame, frame_size), norm_max, norm_min)
    return colormap(resized_normalized_frame)


def blend_many_frames(frames_grey, frames_color):
    blended_frames = []
    for f1, f2 in zip(frames_grey, frames_color):
        blended_frames.append(blend_frames(f1, f2))
    return blended_frames


def blend_frames(fg, fc):
    alpha_factor = np.repeat(fc[:, :, 3:4], 4, axis=2)
    blended_frames = fg*(1-alpha_factor) + fc*alpha_factor
    blended_frames = (blended_frames*255).astype('uint8')
    return blended_frames


def norm_max_min(frame, max_val=None, min_val=None):
    if max_val is None and min_val is None:
        max_val = frame.max()
        min_val = frame.min()
    return (frame - min_val) / (max_val - min_val)


def plot_cam_gif(cam, dir_to_save, save_frame_by_frame=False):

    """ Return animated gif for cam (frame-by-frame)
    Args:

        cam_dict: output of class_activation_maps() i.e.  dictionary

        with level and class as keys (cams as values)

       dir_to_save: directory to save gifs

        save_frame_by_frame: option to save each frame individually

        as a png

    """

    schema = list(cam_dict.keys())
    ncols = 4
    rows_needed = int(len(schema) / ncols)
    if len(schema) % ncols == 0 and rows_needed != 0:
        nrows = rows_needed
    else:
        nrows = rows_needed + 1
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    ax = ax.flatten()
    def image_frame(frame_ind):
        anim = []
        for class_ind, class_id in enumerate(schema):
            ax_x = int(class_ind / ncols)
            ax_y = class_ind % ncols
            index = ncols*ax_x + ax_y
            cam_color = cam_dict[level_to_use][class_id]
            fig.suptitle(f'Class Activation Map Frame: {frame_ind}')
            ax[index].set_title(
                f'\nClass:{schema[class_ind]}')
            ax[index].set_xticks([]), ax[index].set_yticks([])
            anim.append(ax[index].imshow(
                cam_color[frame_ind], animated=True))
            if save_frame_by_frame:
                fig.savefig(os.path.join(
                    dir_to_save,
                    f'cam_level_{level_to_use}_frame_{frame_ind}.png'))
        return anim
    animator = animation.FuncAnimation(
        fig=fig,
        func=image_frame,
        frames=19, blit=True)
    animator.save(
        os.path.join(dir_to_save, f'cam_level_{level_to_use}.gif'))
    plt.close()