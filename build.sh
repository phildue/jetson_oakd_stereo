#/bin/bash
IMAGE_NAME=phildue/vslam-jetson-minimal
docker build . --target developer -t $IMAGE_NAME:dev \
--build-arg UID=$(id -u)
docker build . --target runtime -t $IMAGE_NAME:runtime \
--build-arg UID=$(id -u)