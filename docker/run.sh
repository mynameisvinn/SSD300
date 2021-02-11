source image_defs.sh
PORT="10050"
docker run --rm -it  \
 -p $PORT:$PORT \
 ${IMAGE_NAME}:${IMAGE_VERSION} /bin/bash 
