# Load the name and version of the Docker Image from a common file:
source image_defs.sh

docker build -t ${IMAGE_NAME}:${IMAGE_VERSION} -f ./Dockerfile ..
