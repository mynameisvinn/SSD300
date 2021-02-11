import os
import argparse
import timeit
from ssd300.train_eval import trainer
import logging
import absl.logging

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

LOG = logging.getLogger(__name__)


def parser():
    p = argparse.ArgumentParser(description='Train detection model')

    p.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))  # https://github.com/aws/sagemaker-tensorflow-training-toolkit/issues/340
    p.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))  #!
    p.add_argument('--data_dir', type=str, default = os.environ.get('SM_CHANNEL_TRAINING'), help='directory to store checkpointed models')  #!
    p.add_argument('-o', '--output_path', default = os.environ.get('SM_MODEL_DIR'), help='dir path to write output results', type=str)  #!


    p.add_argument('-g',
                 '--gpu_id',
                 default='',
                 help='provide gpu id (default CPU).',
                 type=str)
    p.add_argument('-e',
                 '--exp_name',
                 default='experiment',
                 help='provide unique experiment name',
                 type=str)
    p.add_argument('-d',
                 '--data_def_dir',
                 required=True,
                 help='Full path .csv data definitions',
                 type=str)
    p.add_argument('--image_npy',
                 default=None,
                 help='Full path to image npy',
                 type=str)
    p.add_argument('--append_data_path',
                   default=None,
                   help='path to npy to append old images to a new npy',
                   type=str)
    p.add_argument('--reload_data_path',
                   default=None,
                   help='Reload saved out data',
                   type=str)
    p.add_argument('--reload_model_path',
                   default=None,
                   help='Reload model weights',
                   type=str)
    p.add_argument('--model_type',
                   default='toothid',
                   help='Choose from [\'dental-materials\', \'tooth-id\']',
                   type=str)
    p.add_argument('--epochs',
                   default=200,
                   help='Number of epochs',
                   type=int)
    p.add_argument('--steps_per_epoch',
                   default=300,
                   help='Steps per epoch',
                   type=int)
    p.add_argument('-b',
                   '--batch_size',
                   default=8,
                   help='Train batch size',
                 type=int)
    p.add_argument('--lr_decay',
                   default=0.1,
                   help='Initial learning rate decay',
                   type=float)
    p.add_argument('--lr_rate',
                   default=100,
                   help='Number of epochs to decay learning rate',
                   type=int)
    p.add_argument('--min_lr',
                   default=0.000001,
                   help='Minimum LR',
                   type=float)
    p.add_argument('--start_lr',
                   default=0.001,
                   help='Starting LR',
                   type=float)
    p.add_argument('--auto_lr',
                   action='store_true',
                   help='Automatically decay learning rate')
    p.add_argument('--balanced_batch',
                   action='store_true',
                   help='Balance batch during training')
    p.add_argument('--mislabel_prob',
                   default=None,
                   type=float)
    p.add_argument('--jitter_prob',
                   default=None,
                   type=float)
    p.add_argument('--bbox_style',
                   default='yolo',
                   help='style bbox that is in the csv data',
                   type=str)
    p.add_argument('--data_aug_params_json',
                   default=None,
                   help='filepath to data augmentation params',
                   type=str)
    p.add_argument('--eval_file',
                   default=None,
                   help='path to train and test npy file with list of ids for previous version',
                   type=str)
    p.add_argument('--neptune_file',
                   default=None,
                   help='filepath to neptune token',
                   type=str)
    p.add_argument('-l', '--logging', default='INFO', help='logging level')
    return p.parse_args()

if __name__ == '__main__':

    

    args = parser()

    args.model_dir = os.environ.get('SM_MODEL_DIR')  #!
    args.data_dir = os.environ.get('SM_CHANNEL_TRAINING')  #!
    args.output_path = os.environ.get('SM_MODEL_DIR')

    numeric_log_level = getattr(logging, args.logging.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: choose from {INFO, DEBUG, WARN}.')
    logging.basicConfig(level=numeric_log_level,
                      format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    start_train_time = timeit.default_timer()
    LOG.info('Initializing training module...')
    train_module = trainer.Train(args)
    train_module.train()
    end_train_time = timeit.default_timer()
    train_time = (end_train_time - start_train_time) / 3600
    LOG.info(
          f'Total score time: {train_time} minutes.')
    LOG.info('Train Done.')
