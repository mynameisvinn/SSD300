import  os, json, pickle
import numpy as np

from ..models import model_tools
from ..data_generator import data
from ..eval_utils import visuals
from ..eval_utils import eval_tools
import neptune

import timeit
import logging
LOG = logging.getLogger(__name__)


def choose_id_ind(arr_length, num_rows):
    while True:
        choose_id_ind = np.random.choice(arr_length, num_rows, replace=False)
        yield choose_id_ind


class Eval():

    def __init__(self, args=None):

        if args is None:
            self.args = None
            self.config = None
            self.output_path = None
            self.model = None
            self.data = None
            self.label_index = None
            self.image_ids = None
            self.n_classes = None
            self.neptune_logger = None
            return
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        os.environ['NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE'] = 'True'
        # Establish output path
        if not os.path.exists(os.path.join(args.output_path, 'plots')):
            os.makedirs(os.path.join(args.output_path, 'plots'))
        self.config = json.load(open(
            os.path.join(args.train_output_path, 'config.json')))
        self.output_path = args.output_path
        # Setup data
        LOG.info('Loading eval image and label data...')
        self.data = {}
        if args.eval_npy is None:
            LOG.info('No eval data provided, fetching from training run...')
            data_path = self.config['data_path']
            id_path = os.path.join(args.train_output_path, 'test_ids.npy')
            self.data, test_ids = data.create_id_dataset(data_path, id_path)
            self.image_ids = test_ids
        else:
            LOG.info('Eval data provided, reading data...')
            self.data = np.load(args.eval_npy, allow_pickle=True)[()]
            self.image_ids = None
        self.label_index = {int(v):k for k, v in self.data['label_dict'].items()}
        if args.label_dict is not None:
            labelA = json.load(open(args.label_dict))
            labelB = self.data['label_dict']
            self.label_convert = data.class_label_convert(labelA, labelB)
        else:
            self.label_convert = None
        LOG.info('Load data complete')
        # Setup model
        LOG.info('Loading ssd300 model...')
        self.n_classes = args.n_classes if args.n_classes else len(self.data['label_dict'])
        self.model = model_tools.create_ssd300(self.n_classes, 'evaluation')
        if args.checkpoint_path is None:
            checkpoint_folder = os.path.join(args.train_output_path, 'checkpoints')
            model_tools.load_best_checkpoint(self.model, checkpoint_folder)
        else:
            self.model.load_weights(args.checkpoint_path, by_name=True)
        LOG.info(self.model.summary())
        # Setup neptune
        if args.neptune_file is not None:
            with open(args.neptune_file, 'r') as f:
                token = f.read()
                model_type = self.config['model_type']
                project = neptune.init(f'kells/{model_type}', api_token=token)
                self.neptune_logger = project.get_experiments(id=self.config['neptune_id'])[0]
        else:
            self.neptune_logger = None

    def evaluate(self, plot_visuals=True, save=True, max_num_to_plot=10):
        LOG.info('Begin evaluation...')
        apply_unique_box = self.config['model_type'] in ['tooth-id']
        apply_adjacent_rule = False
        evaluate_obj = eval_tools.get_evaluate_func(
            model=self.model,
            n_classes=len(self.label_index),
            label_convert=self.label_convert,
            apply_unique_box=apply_unique_box,
            apply_adjacent_rule=apply_adjacent_rule)
        predict_generator = data.create_generator(x=self.data['images'],
                                                  y=self.data['labels'],
                                                  image_ids=self.image_ids,
                                                  label_encoder=None)
        evaluator, results = evaluate_obj(predict_generator, len(self.data['images']))
        eval_results = {'metrics': results,
                        'predictions': evaluator.prediction_results,
                        'num_gt_per_class': evaluator.num_gt_per_class,
                        'true_positives': evaluator.true_positives,
                        'false_positives': evaluator.false_positives,
                        'false_negatives': evaluator.false_negatives}
        if save:
            np.save(os.path.join(self.output_path, 'eval_results.npy'), eval_results)
        if plot_visuals:
            visuals.plot_average_precision(results=eval_results['metrics'],
                                           label_index=self.label_index,
                                           save_path=os.path.join(self.output_path, 'plots'),
                                           neptune_logger=self.neptune_logger)
            stats = eval_tools.compute_recall_precision_curves(
                class_num_gt=eval_results['num_gt_per_class'],
                true_positives=eval_results['true_positives'],
                false_positives=eval_results['false_positives'],
                prediction_results=eval_results['predictions'])
            visuals.plot_RC_curve(stats=stats,
                                  label_index=self.label_index,
                                  save_path=os.path.join(self.output_path, 'plots'),
                                  neptune_logger=self.neptune_logger)
            image_labels = eval_tools.process_results_per_image(eval_results['predictions'])
            ind_id_to_plot = eval_tools.identify_missed_detections(image_labels, self.image_ids, self.data['labels'])
            num_rows =  min(len(ind_id_to_plot), 5)
            num_plots = min(int(len(ind_id_to_plot) / num_rows) + 1, max_num_to_plot)
            for i in range(0, num_plots):
                chosen = next(choose_id_ind(len(ind_id_to_plot), num_rows))
                pass_id_ind = [ind_id_to_plot[cc] for cc in chosen]
                visuals.plot_sample_results(image_labels=image_labels,
                                           test_x=self.data['images'],
                                           test_y=self.data['labels'],
                                           ind_id_to_plot=pass_id_ind,
                                           label_index=self.label_index,
                                           save_path=os.path.join(self.output_path, 'plots', f'miss_detections_plot_{i}.png'),
                                           neptune_logger=self.neptune_logger)
        return evaluator, results
