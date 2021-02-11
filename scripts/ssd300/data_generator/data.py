import os
from urllib import request
import tempfile
import ssl

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from ..models import model_tools

import logging
LOG = logging.getLogger(__name__)


def create_label(labels, label_dict):
    return [label_dict[ll] for ll in labels]


def check_boundaries(bbox, img_width, img_height):
    bbox = bbox.copy()
    bbox[bbox < 0] = 0
    bbox[0] = bbox[0] if bbox[0] < img_width else (img_width - 1)
    bbox[1] = bbox[1] if bbox[1] < img_height else (img_height - 1)
    bbox[2] = bbox[2] if bbox[2] < img_width else img_width
    bbox[3] = bbox[3] if bbox[3] < img_height else img_height
    return bbox


def fix_bad_bounding_box(bbox):
    """Zero out bounding boxes if it's negative"""
    """Make sure boxes make sense. Assumes ssd style"""
    bbox = bbox.copy()
    if bbox[2] <= bbox[0]:
        bbox[2] = bbox[0] + 1
    if bbox[3] <= bbox[1]:
        bbox[3] = bbox[1] + 1
    return bbox


def bbox_style(bbox, style='ssd'):
    """Option to use yolo style bbox"""
    if style == 'ssd':
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
    elif style == 'yolo':
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
    else:
        raise Exception('Invalid style for bounding box')

    return bbox


def process_bbox(bboxes, change_style=None):
    new_bboxes = []
    for bbox in bboxes:
        if change_style is not None:
            bbox = bbox_style(bbox, style=change_style)
        new_bboxes.append(bbox)
    return new_bboxes


def mislabel(ylabel, n_classes, prob=0.05):
    ylabel = ylabel.copy()
    if np.random.rand() < prob:
        ylabel[:, 0] = np.random.randint(
            low=0, high=n_classes, size=ylabel.shape[0])
    return ylabel


def jitter(ylabel, img_width, img_height, prob):
    ylabel = ylabel.copy()
    length = int(min(img_width, img_height) * 0.05)
    if np.random.rand() < prob:
        for ll in ylabel:
            ll[1] = ll[1] + np.random.randint(-length, length)
            ll[2] = ll[2] + np.random.randint(-length, length)
            ll[3] = ll[3] + np.random.randint(-length, length)
            ll[4] = ll[4] + np.random.randint(-length, length)
    return ylabel


def class_label_convert(label_from, label_to):
    # standardize format string key and int value
    label_from = {str(k): int(v) for k, v in label_from.items()}
    label_to = {str(k): int(v) for k, v in label_to.items()}
    label_convert = {}
    for class_name, class_id in label_from.items():
        if class_name in label_to:
            label_convert[class_id] = label_to[class_name]
        else:
            label_convert[class_id] = 0
    return label_convert


def label_convert(labels, label_from, label_to):
    convert_dict = class_label_convert(label_from, label_to)
    for label_id, label_set in labels:
        label_set[:, 0] = create_label(label_set[:, 0], convert_dict)


def combine(dataA, dataB):
    id_A, item_A = zip(*dataA)
    id_B, item_B = zip(*dataB)
    id_A = list(id_A)
    item_A = list(item_A)
    for ind, image_id in enumerate(id_B):
        if image_id not in id_A:
            id_A.append(image_id)
            item_A.append(item_B[ind])
    return list(zip(id_A, item_A))


def combine_datasets(datasetA, datasetB):
    # Resolve any label conflicts
    label_convert(labels=datasetB['labels'],
                  label_from=datasetB['label_dict'],
                  label_to=datasetA['label_dict'])
    # Combine images and labels
    images = combine(datasetA['images'], datasetB['images'])
    labels = combine(datasetA['labels'], datasetB['labels'])
    return {'images': images,
            'labels': labels,
            'label_dict': datasetA['label_dict']}


def read_csvs_into_pd(csv_paths, return_def=False):
    data_def ={i.split('.')[0]: pd.read_csv(os.path.join(csv_paths, i)) for i in os.listdir(csv_paths) if i.endswith('csv')}
    data_def['bbox_labels'] = data_def['bbox_labels'].rename({'id': 'bbox_id'}, axis=1)
    data_def['annotators'] = data_def['annotators'].rename(
        {'id': 'annotator_id', 'name': 'annotator_name',
         'labeling_platform_id': 'annotator_email', 'gt': 'ground_truth'}, axis=1)
    data_def['projects'] = data_def['projects'].rename({'id': 'project_id', 'name': 'project_name'}, axis=1).drop(
        ['labeling_platform_id'], axis=1)
    data_def['bbox_attributes'] = data_def['bbox_attributes'].rename(
        {'id': 'attribute_id', 'name': 'label_type', 'value': 'label'}, axis=1)
    data_def['images'] = data_def['images'].rename({'id': 'image_id'}, axis=1)  
    df = data_def['bbox_labels'].merge(
       data_def['annotators'], left_on='annotator_id', right_on='annotator_id', how='left').merge(
       data_def['bbox_relations'], left_on='bbox_id', right_on='bbox_id', how='left').merge(
       data_def['bbox_attributes'], left_on='attribute_id', right_on='attribute_id', how='left').merge(
       data_def['images'], left_on='image_id', right_on='image_id', how='left').merge(
       data_def['projects'], left_on='project_id', right_on='project_id', how='left')
    df = df.sort_values(by=['storage_id', 'bbox_id'])
    df.loc[df[ 'annotator_id'].isna(), 'annotator_id'] = 'xx'
    df = df[~df.label.isna()]
    df = df[~df.storage_id.isna()]
    df['bbox'] = df[['x_min', 'y_min', 'width', 'height']].values.tolist()
    if return_def:
        return df, data_def
    return df


def load_data(data_def_dir, convert_to_ssd=False, image_npy=None, append_data_path=None):
    data_df = read_csvs_into_pd(data_def_dir)
    id_url_pair = data_df[['storage_id', 'access_url']].drop_duplicates().values.tolist()
    storage_ids, urls = zip(*id_url_pair)
    labels, label_dict = load_labels(data_df, storage_ids, convert_to_ssd=convert_to_ssd)
    append_id = []
    if append_data_path is not None:
        append_npy = np.load(append_data_path, allow_pickle=True)[()]
        append_id = set([i for i, j in append_npy['images']])
    if image_npy is None:
        images = load_images(id_url_pair, append_id=append_id)
    else:
        LOG.info('Loading data from npy...')
        images = np.load(image_npy, allow_pickle=True)[()]
    dataset = {'images':images, 'labels': labels, 'label_dict': label_dict}
    if append_data_path is not None:
        dataset = combine_datasets(dataset, append_npy)
    if len(dataset['images']) != len(dataset['labels']):
        image_len = len(dataset['images'])
        label_len = len(dataset['labels'])
        raise Exception(f'Misaligned datasets - images: {image_len} labels: {label_len}')
    return dataset


def load_images(id_url_pair, append_id=[]):
    context = ssl._create_unverified_context()
    LOG.info(f'Reading and loading {len(id_url_pair)} images...')
    image_data = []
    total_images = len(id_url_pair)
    for i, (storage_id, image_url) in enumerate(id_url_pair):
        LOG.info(f'Reading {i} of {total_images}: {image_url}')
        if not storage_id in append_id: # Avoid reading the same image that already exists
            image_data.append((storage_id, read_image(image_url, context)))
        else:
            LOG.info('Image in append dataset...skip reading')
    return image_data


def read_image(url, context):
    LOG.info(f'Reading {url}')
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
        f.write(request.urlopen(url, context=context).read())
        img=mpimg.imread(tmp.name)
    return np.array(img)
        

def load_labels(data_df, img_id, convert_to_ssd=True):
    label_category = pd.unique(data_df.label).astype('object')
    label_category.sort()
    label_dict = {lc: (i + 1) for i, lc in enumerate(label_category)} #reserve 0 for the background class
    LOG.info(f'Label categories: {label_dict}')
    LOG.info('Processing and loading labels...')
    label_data = []
    for iid in img_id:
        aid = data_df[data_df.storage_id.eq(iid)].annotator_id.values[0]
        uniq = data_df[data_df.storage_id.eq(iid) & data_df.annotator_id.eq(aid)].drop_duplicates(['bbox_id'])
        label = create_label(uniq.label.astype('object').values.tolist(), label_dict)
        bbox = process_bbox(uniq.bbox.values.tolist(), change_style='ssd' if convert_to_ssd else None)
        if len(label) == 0 or len(bbox) == 0:
            label_data.append((iid, []))
            continue
        combine = np.hstack([np.array(label).reshape(-1, 1), bbox])
        label_data.append((iid, combine))
    return label_data, label_dict


def convert_id_with_data(current_match, match_df, from_id, to_id):
    current_ids, current_data = zip(*current_match)
    new_ids = convert_id(current_ids, match_df, from_id, to_id)
    return list(zip(new_ids, current_data))


def convert_id(current_ids, match_df, from_id, to_id):
    from_ids, to_ids = zip(*match_df[[from_id, to_id]].drop_duplicates().values.tolist())
    lookup = {str(fid):n for n, fid in enumerate(from_ids)}
    new_ids = [to_ids[lookup[str(cid)]] for cid in current_ids]
    return new_ids


def grab_ids(id_image_pair, get_ids):
    try:
        data_ids, core_data = zip(*id_image_pair)
        lookup_id = {str(idd):u for u, idd in enumerate(data_ids)}
        return [core_data[lookup_id[str(i)]] for i in get_ids]
    except KeyError:
        raise Exception('ID was not found in the data pair')


def create_id_dataset(data_path, id_path):
    id_dataset = {}
    dataset = np.load(data_path, allow_pickle=True)[()]
    ids_to_choose = np.load(id_path, allow_pickle=True)[()]
    id_dataset['images'] = grab_ids(dataset['images'], ids_to_choose)
    id_dataset['labels'] = grab_ids(dataset['labels'], ids_to_choose)
    id_dataset['label_dict'] = {str(ll): ii for ll, ii in dataset['label_dict'].items()}
    return id_dataset, ids_to_choose


class create_generator():

    def __init__(self, x, y, image_ids, label_encoder=None):
        self.images = x
        self.labels = y
        self.image_ids = image_ids
        self.label_encoder = label_encoder
        self.resize = model_tools.resize()      
        self.convert = model_tools.convert_to_3_channels()
        self.eval_neutral = None

    def get_dataset_size(self):
        return len(self.images)

    def generate(self, batch_size, shuffle=False, transformations=None, label_encoder=None,
                 returns={'processed_images', 'image_ids', 'evaluation-neutral', 'inverse_transform', 'original_labels'},
                 keep_images_without_gt=True, degenerate_box_handling='remove'):
        while True:
            if batch_size is None:
                batch_size = len(self.images)
            indices = np.random.choice(np.arange(len(self.images)), batch_size, replace=False)
            orig_x = []
            orig_y = []
            batch_x = []
            batch_y = []
            batch_id = []
            inverter = []
            for i in indices:
                xr = self.images[i].copy()
                yr = self.labels[i].copy()
                orig_x.append(self.images[i])
                orig_y.append(self.labels[i])
                it = []
                if self.image_ids is not None:
                    im_id = self.image_ids[i]
                else:
                    im_id = i
                xr = self.convert(xr)
                xr, yr, inverse_trans = self.resize(xr, yr, return_inverter=True)
                it.append(inverse_trans)
                yr[:, 3] = yr[:,3] + 1
                yr[:, 4] = yr[:,4] + 1
                batch_x.append(xr)
                batch_y.append(yr)
                batch_id.append(im_id)
                inverter.append(it)
            if label_encoder is not None:
                batch_y = label_encoder(batch_y, diagnostics=False)
            ret = []
            if 'processed_images' in returns:
                ret.append(np.array(batch_x))
            if 'processed_labels' in returns:
                ret.append(batch_y)
            if 'image_ids' in returns:
                ret.append(batch_id)
            if 'evaluation-neutral' in returns:
                ret.append(None)
            if 'inverse_transform' in returns:
                ret.append(inverter)
            if 'original_images' in returns:
                ret.append(orig_x)
            if 'original_labels' in returns:
                ret.append(orig_y)
            yield ret
