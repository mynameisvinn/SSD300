import os
import numpy as np
import argparse
from ssd300.data_generator import data
import logging
import absl.logging

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

LOG = logging.getLogger(__name__)


def parser():
    p = argparse.ArgumentParser(description='Data ID conversion tool')
    p.add_argument('--data_file_to_convert', default=None, help='data npy')
    p.add_argument('--data_file_type', required=True, help='data type')
    p.add_argument('--data_def', required=True, help='data definition for converting id')
    p.add_argument('--from_id', required=True, help='current id')
    p.add_argument('--to_id', required=True, help='new id to convert to')
    p.add_argument('--output_path', required=True, help='location to save converted data')
    p.add_argument('-l', '--logging', default='INFO', help='logging level')
    return p.parse_args()

if __name__ == '__main__':
    args = parser()
    numeric_log_level = getattr(logging, args.logging.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: choose from {INFO, DEBUG, WARN}.')
    logging.basicConfig(level=numeric_log_level,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    LOG.info('Converting data...')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    basename = os.path.basename(args.data_file_to_convert)
    write_file = os.path.join(args.output_path, basename)
    data_to_change = np.load(args.data_file_to_convert, allow_pickle=True)[()]
    data_def = data.read_csvs_into_pd(args.data_def)
    if args.data_file_type == 'data':
        data_to_change['images'] = data.convert_id_with_data(data_to_change['images'], data_def, args.from_id, args.to_id)
        data_to_change['labels'] = data.convert_id_with_data(data_to_change['labels'], data_def, args.from_id, args.to_id)
    elif args.data_file_type == 'ids':
        data_to_change = data.convert_id(data_to_change, data_def, args.from_id, args.to_id)
    else:
        raise Exception('Invalid data file type: only choose \'data\' or \'ids\'')
    np.save(write_file, data_to_change)
    LOG.info('Conversion Done.')
