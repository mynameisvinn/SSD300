import argparse
from sagemaker.tensorflow import TensorFlow
import sagemaker
import os


def parser():
    p = argparse.ArgumentParser(description='Train detection model')
    p.add_argument('--mode', type=str, required=True)
    return p.parse_args()

def main(mode: str):
    sess = sagemaker.Session()
    role = "SageMakerRole"

    tf_estimator = TensorFlow(entry_point='train_ssd300.py', 
                              role=role,
                              source_dir="scripts",
                              instance_count=1, 
                              instance_type=mode,
                              framework_version='1.12.0', 
                              py_version='py3',
                              script_mode=True,
                              dependencies=['scripts/ssd300'],
                              hyperparameters={
                                  'epochs': 5,
                                  'batch_size': 1,
                                  'data_dir': '/opt/ml/input/data/training',
                                  'data_def_dir': '/opt/ml/input/data/training/tooth_id_v1.3',
                                  'reload_data_path': '/opt/ml/input/data/training/image_label_sample_data.npy',
                                  'exp_name': 'myexperiment',
                                  'model_type': 'tooth-id',
                                  'steps_per_epoch': 1,
                                  'model_dir': '/opt/ml/model'
                              }
                             )
    data_dir = os.path.join(os.getcwd(), 'for_vin')
    f'file://{data_dir}'

    inputs = {'training': f'file://{data_dir}'}
    tf_estimator.fit(inputs) 


if __name__ == '__main__':
    args = parser()
    main(args.mode)
