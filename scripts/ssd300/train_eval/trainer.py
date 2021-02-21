import os, json
import numpy as np, pandas as pd

import timeit
import logging
LOG = logging.getLogger(__name__)

from ..models import model_tools
from ..data_generator import data
import neptune


def create_train_test_split(data_dict, train_ratio=0.75, eval_file=None):
    image_ids, _ = zip(*data_dict['images'])
    num_sample = int(train_ratio*len(data_dict['images']))
    if eval_file is not None:
        white_list = np.load(os.path.join(eval_file, 'train_ids.npy'), allow_pickle=True)[()]
        black_list = np.load(os.path.join(eval_file, 'test_ids.npy'), allow_pickle=True)[()]
    else:
        white_list = []
        black_list = []
    init_train_ids = np.intersect1d(image_ids, white_list)
    if len(init_train_ids) < num_sample:
        num_sample = num_sample - len(init_train_ids)
        candidates = np.setdiff1d(image_ids, init_train_ids)
        candidates = np.setdiff1d(candidates, black_list)
        if num_sample > len(candidates):
            num_sample = len(candidates)
        remaining_train_ids = np.random.choice(candidates, num_sample, replace=False)
        train_ids = np.union1d(init_train_ids, remaining_train_ids)
    else:
        train_ids = init_train_ids
    test_ids = np.setdiff1d(image_ids, train_ids)
    train = {}
    train['images'] = data.grab_ids(data_dict['images'], train_ids)
    train['labels'] = data.grab_ids(data_dict['labels'], train_ids)
    test = {}
    test['images'] = data.grab_ids(data_dict['images'], test_ids)
    test['labels'] = data.grab_ids(data_dict['labels'], test_ids)
    return train, test, train_ids, test_ids


def create_data_generator(n_classes,
                          use_resize=False,
                          convert_3_channel=False,
                          data_aug=False,
                          data_aug_params=None,
                          ssd_input_encoder=None,
                          mislabel_prob=None,
                          jitter_prob=None):
    data_augmentation = model_tools.data_aug(data_aug_params)
    resize = model_tools.resize()
    convert = model_tools.convert_to_3_channels()

    def generator(x, y, y_inverse=None, y_prob=None, batch_size=1):
        while True:
            if y_inverse is not None and y_prob is not None:
                y_chosen = np.random.choice(np.arange(len(y_prob)), batch_size, p=y_prob)
                indices = [np.random.choice(y_inverse[cc], 1)[0] for cc in y_chosen]
            else:
                indices = np.random.choice(np.arange(len(x)), batch_size, replace=False)
            batch_x = []
            batch_y = []
            for i in indices:
                xi = x[i]
                yi = y[i]
                assert not np.any(np.isnan(xi))
                assert not np.any(np.isnan(yi))
                if convert_3_channel:
                    xi = convert(xi)
                if data_aug:
                    xi, yi = data_augmentation(xi, yi)
                if use_resize:
                    xi, yi = resize(xi, yi)
                if mislabel_prob is not None:
                    yi = data.mislabel(yi, n_classes=n_classes, prob=mislabel_prob)
                if jitter_prob is not None:
                    yi = data.jitter(yi, img_width=xi.shape[0],
                        img_height=xi.shape[1], prob=jitter_prob)
                for ll in yi:
                    ll[1:] = data.check_boundaries(ll[1:], xi.shape[0], xi.shape[1])
                    ll[1:] = data.fix_bad_bounding_box(ll[1:])
                batch_x.append(xi)
                batch_y.append(yi)
            if ssd_input_encoder is not None:
                batch_y = ssd_input_encoder(batch_y, diagnostics=False)
            yield np.array(batch_x), batch_y
    return generator


def balance_prob(train_y, label_dict):
    train_y_inverse = {v:[] for v in label_dict.values()}
    for i, arr in enumerate(train_y):
        if len(arr) == 0:
            continue
        for c in arr[:, 0]:
            train_y_inverse[c].append(i)
    arr_form = np.array([v for v in train_y_inverse.values()])
    sum_y = np.array([len(v) for v in arr_form])
    total = sum_y.sum()
    prob_y = sum_y/total
    pya = prob_y.argsort()
    reverse_y_prob = prob_y[pya[np.flipud(pya).argsort()]]
    return arr_form, reverse_y_prob


class Train():

    def __init__(self, args):
        self.args = args
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        os.environ['NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE'] = 'True'
        # Establish output path
        if not os.path.exists(args.output_path):
            os.makedirs(os.path.join(args.output_path,
                                     'checkpoints'))
        self.output_config = {}
        self.output_config['exp_name'] = args.exp_name
        self.output_config['model_type'] = args.model_type
        # Setup data
        LOG.info('Loading image and label data...')
        if args.reload_data_path is None:
            LOG.info('No pre-existing data available, fetching data...')
            self.data = data.load_data(args.data_def_dir,
                                       convert_to_ssd=(args.bbox_style != 'ssd'),
                                       image_npy=args.image_npy,
                                       append_data_path=args.append_data_path)
            saved_data_path = os.path.join(args.output_path, 'image_label_data.npy')
            LOG.info('Saving data...')
            np.save(saved_data_path, self.data)
            self.output_config['data_path'] = saved_data_path
        else:
            LOG.info('Pre-existing data available, reading data...')
            self.data = np.load(args.reload_data_path, allow_pickle=True)[()]
            self.output_config['data_path'] = args.reload_data_path
            if args.append_data_path is not None:
                LOG.info('Appending data available, reading data...')
                append_npy = np.load(args.append_data_path, allow_pickle=True)[()]
                self.data = data.combine_datasets(self.data, append_npy)
                saved_data_path = os.path.join(args.output_path, 'image_label_data.npy')
                LOG.info('Saving appended data...')
                np.save(saved_data_path, self.data)
                self.output_config['data_path'] = saved_data_path
        self.output_config['label_dict'] = self.data['label_dict']
        json.dump(self.data['label_dict'],
            open(os.path.join(args.output_path, 'label_dict.json'), 'w'))
        LOG.info('Load data complete')   
        # Setup model
        LOG.info('Loading ssd300 model...')
        self.n_classes = len(self.data['label_dict'])
        if args.reload_model_path is None:
            self.model = model_tools.create_ssd300(
                self.n_classes, 'training', args.start_lr)
        else:
            self.model = model_tools.reload_ssd300(args.reload_model_path)
        LOG.info(self.model.summary())

        # Setup neptune
        if args.neptune_file is not None:
            self.neptune_log = True
            with open(args.neptune_file, 'r') as f:
                token = f.read()
                neptune.init(f'kells/{args.model_type}', api_token=token)
                exp = neptune.create_experiment(name=f'{args.exp_name}')
                neptune.log_text(log_name='model', x='ssd300')
                neptune.log_text(log_name='model_config', x=json.dumps(self.model.get_config()))
                neptune.log_text(log_name='optimizer', x='Adam')
                self.output_config['neptune_id'] = exp.id
        else:
            self.neptune_log = False

    def train(self):
        args = self.args
        train, test, train_ids, test_ids = create_train_test_split(
            self.data, eval_file=args.eval_file)
        np.save(os.path.join(args.output_path, 'train_ids.npy'), train_ids)
        np.save(os.path.join(args.output_path, 'test_ids.npy'), test_ids)
        if args.balanced_batch:
            y_inverse, y_prob = balance_prob(train['labels'], self.data['label_dict'])
        else:
            y_inverse, y_prob = None, None
        ssd_input_encoder = model_tools.create_ssd_encoder(self.model, self.n_classes)
        if args.data_aug_params_json is not None:
            data_aug_params = json.load(open(args.data_aug_params_json))
        else:
            data_aug_params = None
        train_generator = create_data_generator(n_classes=self.n_classes,
                                                use_resize=False,
                                                convert_3_channel=False,
                                                data_aug=True,
                                                data_aug_params=data_aug_params,
                                                ssd_input_encoder=ssd_input_encoder,
                                                mislabel_prob=args.mislabel_prob,
                                                jitter_prob=args.jitter_prob)
        test_generator = create_data_generator(n_classes=self.n_classes,
                                               use_resize=True,
                                               convert_3_channel=True,
                                               ssd_input_encoder=ssd_input_encoder)
        callbacks = model_tools.train_callbacks(
            args.output_path,
            args.auto_lr,
            args.start_lr,
            args.lr_decay,
            args.lr_rate,
            args.min_lr,
            args.epochs,
            self.neptune_log)
        history = self.model.fit_generator(
            generator=train_generator(
                x=train['images'],
                y=train['labels'],
                y_inverse=y_inverse,
                y_prob=y_prob,
                batch_size=args.batch_size),
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            initial_epoch=0,
            callbacks=callbacks,
            validation_data=test_generator(
                x=test['images'],
                y=test['labels'],
                batch_size=args.batch_size),
            validation_steps=int(len(test['images']) / args.batch_size))
        json.dump(self.output_config,
                  open(os.path.join(self.args.output_path, 'config.json'), 'w'))

        print("savvvvvving")
        LOG.info("savvvvvving")
        self.model.save(os.path.join(os.environ.get('SM_MODEL_DIR'), 'sm_example_model.h5'))  #!
        self.model.save(os.path.join(os.environ.get('MODEL_DIR'), 'm_example_model.h5'))  #!