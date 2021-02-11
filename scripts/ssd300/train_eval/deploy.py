import  os, json, pickle, timeit
import numpy as np
from matplotlib import pyplot as plt, cm

from ..models import model_tools
from ..eval_utils import visuals
import neptune

import timeit
import logging
LOG = logging.getLogger(__name__)


def read_class_dict(filename):
  return {str(k):v for k, v in json.load(open(filename)).items()}


def process_predictions(y_pred,
                        max_height,
                        max_width,
                        apply_unique_box=True,
                        apply_adjacent_rule=True):
  # Max suppression
  results_class = {}
  for k, box in enumerate(y_pred):
    class_id = int(box[0])
    confidence = box[1]
    xmin = max(round(box[2], 1), 0)
    ymin = max(round(box[3], 1), 0)
    xmax = min(round(box[4], 1), max_width)
    ymax = min(round(box[5], 1), max_height)
    processed_box = (class_id, confidence, xmin, ymin, xmax, ymax)
    if class_id not in results_class:
      results_class[class_id] = [processed_box]
    else:
      if apply_unique_box:
        if confidence > results_class[class_id][0][1]:
          results_class[class_id] = [processed_box]
      else:
        results_class[class_id].append(processed_box)
    # Remove box that don't follow adjacency rule
  if apply_adjacent_rule:
    class_keys = list(results_class.keys())
    for c1 in class_keys:
      aj = False
      for c2 in class_keys:
        if ((c1 == (c2 + 1)) or (c1 == (c2 - 1))):
          aj = True
          break
      if not aj and results_class[c1][0][1] < 0.7:
        del (results_class[c1])
  results = sum([vl for cl, vl in results_class.items()],[])
  return results


class Deploy():

    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        os.environ['NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE'] = 'True'
        if not os.path.exists(args.label_dict):
              raise Exception(f'Cannot locate the class dictionary: {args.label_dict}')
        else:
              self.label_dict = read_class_dict(args.label_dict)
        self.label_index = {int(v):k for k, v in self.label_dict.items()}
        LOG.info('Loading ssd300 model...')
        self.model = model_tools.create_ssd300(len(self.label_dict), 'evaluation')
        if args.checkpoint_path is None:
            checkpoint_folder = os.path.join(args.train_output_path, 'checkpoints')
            model_tools.load_best_checkpoint(self.model, checkpoint_folder)
        else:
            self.model.load_weights(args.checkpoint_path, by_name=True)
        LOG.info(self.model.summary())



    def score_image(self, img_arr, apply_unique_box, apply_adjacent_rule):
      """Predict on an image array.

      Args:
        model: loaded model object.
        img_arr: RGB image as numpy array.
      Returns:
        prediction results.
      """
      # prepare input image.
      resize = model_tools.resize()
      convert = model_tools.ConvertTo3Channels()
      labels_format = {'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}
      # These only get applied if model_type is toothid
      img_height, img_width = img_arr.shape[:2]
      to_three_channel = convert(img_arr)
      resized_image, inverter = resize(to_three_channel, return_inverter=True)
      # make prediction.
      predictions = self.model.predict(np.expand_dims(resized_image, 0))[0]
      # post-process predictions.
      y_pred_filtered = []
      for i in range(len(predictions)):
        if not (predictions[i] == 0).all():
          y_pred_filtered.append(predictions[i])
      if len(y_pred_filtered) > 0:
        y_pred = inverter(y_pred_filtered)
        y_pred = process_predictions(
          y_pred, img_height, img_width,
          apply_unique_box, apply_adjacent_rule)
      else:
        y_pred = []
      res = {'img_array': img_arr, 'pred': y_pred}
      return res


    def save_image(self, save_dir, result):  
        filename = result['filename']
        basename = os.path.basename(filename).split('.')[0]
        write_filename = os.path.join(save_dir, basename + '_results.png')
        LOG.info(f'Plotting and writing image file: {write_filename}')
        visuals.plot_image(
            image=result['img_array'],
            label=result['pred'],
            label_index=self.label_index,
            style='ssd',
            title=basename,
            save_path=write_filename,
            label_format={'class_id': 0, 'score': 1, 'x1': 2, 'y1': 3, 'x2': 4, 'y2': 5})