#! /bin/bash

IMAGE_NAME="ml_dental_ssd300"
IMAGE_VERSION="1.0.0"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export VERSION_NUMBER=$IMAGE_VERSION
export IMAGE_NAME=$IMAGE_NAME
