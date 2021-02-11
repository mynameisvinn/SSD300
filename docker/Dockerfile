FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

WORKDIR /home/kells/dev

RUN apt-get update -y
RUN apt-get install -y vim wget
RUN apt-get install -y libgtk2.0-dev libsm6 libxext6 libxrender-dev

# Install pip for python
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

COPY . ./ml_dental_ssd300/

RUN pip install -r ml_dental_ssd300/requirements.txt
RUN pip install -e ml_dental_ssd300

ENV NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE="True"
