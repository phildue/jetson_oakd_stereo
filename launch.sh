#!/bin/bash

ros2 run depthai_examples stereo_node --ros-args -p camera_model:=OAK-D-LITE &
python3 stereo_network_node.py

killall stereo_node
#killall point_cloud_node