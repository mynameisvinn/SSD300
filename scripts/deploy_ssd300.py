import os
import argparse
import timeit

import cv2
import numpy as np

from ssd300.train_eval import deploy
import logging
import absl.logging

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

LOG = logging.getLogger(__name__)


def parser():
    p = argparse.ArgumentParser(description='Evaluate detection model')
    p.add_argument('-g',
                 '--gpu_id',
                 default='',
                 help='provide gpu id (default i.e. CPU).',
                 type=str)
    p.add_argument('--data_dir',
                   required=True,
                   help='directory path to jpg or png images',
                   type=str)
    p.add_argument('-o',
                 '--output_path',
                 required=True,
                 help='dir to output eval results',
                 type=str)
    p.add_argument('-c',
                   '--checkpoint_path',
                   default=None,
                   help='path to checkpoint weights to load',
                   type=str)
    p.add_argument('-t',
                   '--train_output_path',
                   default=None,
                   help='dir path with trained model',
                   type=str)
    p.add_argument('--label_dict',
                   required=True,
                   help='path to label_dict in json format if label dict is different than the eval data',
                   type=str)
    p.add_argument('--model_type',
                   required=True,
                   help='must be tooth-id or dental-materials',
                   type=str)
    p.add_argument('--plot_results',
                   action='store_true',
                   help='boolean to plot predictions')
    p.add_argument('-l', '--logging', default='INFO', help='logging level')
    return p.parse_args()


if __name__ == '__main__':
    args = parser()
    numeric_log_level = getattr(logging, args.logging.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: choose from {INFO, DEBUG, WARN}.')
    logging.basicConfig(level=numeric_log_level,
                      format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    start_score_time = timeit.default_timer()
    LOG.info('Initializing deployment module...')
    deployment = deploy.Deploy(args)
    apply_unique_box = (args.model_type in ['toothid'])
    apply_adjacent_rule = False
    results = []
    img_list = [os.path.join(args.data_dir, img) for img in os.listdir(args.data_dir)]
    for img_file in img_list:
        try:
            LOG.info(f'Scoring image file: {img_file}')
            start_time = timeit.default_timer()
            img = cv2.imread(img_file)
            res = deployment.score_image(img, apply_unique_box, apply_adjacent_rule)
            res['filename'] = img_file
            results.append(res)
            end_time = timeit.default_timer()
            s_time = (end_time - start_time) / 60
            LOG.info(f'Score time: {s_time} seconds')
        except Exception as ex:
            LOG.debug(str(ex))
    end_score_time = timeit.default_timer()
    score_time = (end_score_time - start_score_time) / 60
    LOG.info(
          f'Total score time: {score_time} seconds for {len(img_list)} images')
    np.save(os.path.join(args.output_path, 'results.npy'), results)
    if args.plot_results:
        plot_path = os.path.join(args.output_path, 'plots')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        for result in results:
            deployment.save_image(plot_path, result)
    LOG.info('Scoring done.')
