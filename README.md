# Kells Dental Single Shot Detector 

Core Python package contents:

 * `scripts/`
 * `ssd300/`
 * `setup.py`
 * `MANIFEST.in`

Container-related contents:

 * `docker`


### Installation
There are two ways to install this package:

1. Docker - build the docker image and run the docker container (`docker/build.sh` and `docker/run.sh`)

With Docker installed in the environment:
```
cd docker
./build.sh && ./run.sh
```

2. Pip install - use a python virtual environment and pip install

With python3.7, venv, and pip installed in the environment
```
python -m venv ssd300
source ssd300/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ssd300

# To deactivate virtual environment
deactivate
```

### Train an ssd300 model

Use the templates in `examples` for running the `train_ssd.py` python script.

Here are the arguments to use for a baseline training run:

exp=experiment_name
batch_size=32  # Only needs to change if architecture is very big.

### Distributing

To make a tarball with working version info, run:

```bash
python setup.py sdist
```

Then collect the tarball from `dist/`.
