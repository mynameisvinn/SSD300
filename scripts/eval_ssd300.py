import argparse
import timeit
from ssd300.train_eval import evaluator
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
    p.add_argument('--n_classes',
                   default=None,
                   help='Provide if number of classes for model is different than data',
                   type=int)
    p.add_argument('-t',
                 '--train_output_path',
                 default=None,
                 help='dir path with trained model',
                 type=str)
    p.add_argument('-e',
                   '--eval_npy',
                   default=None,
                   help='path to numpy pickle with evaluation data.  Default is to use test set from training',
                   type=str)
    p.add_argument('--label_dict',
                   default=None,
                   help='path to label_dict in json format if label dict is different than the eval data',
                   type=str)
    p.add_argument('--neptune_file',
                   default=None,
                   help='filepath to neptune token',
                   type=str)
    p.add_argument('-l', '--logging', default='INFO', help='logging level')
    return p.parse_args()

if __name__ == '__main__':
    args = parser()
    numeric_log_level = getattr(logging, args.logging.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: choose from {INFO, DEBUG, WARN}.')
    logging.basicConfig(level=numeric_log_level,
                      format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    start_eval_time = timeit.default_timer()
    LOG.info('Initializing evaluation module...')
    evaluation = evaluator.Eval(args)
    evaluation.evaluate()
    end_eval_time = timeit.default_timer()
    eval_time = (end_eval_time - start_eval_time) / 60
    LOG.info(
          f'Total evaluation time: {eval_time} seconds.')
    LOG.info('Evaluation done.')
