#!/bin/bash
set -e

# setup ros environment
echo "Sourcing /opt/ros/$ROS_DISTRO/install/setup.bash.."
source "/opt/ros/$ROS_DISTRO/install/setup.bash"
echo "Sourcing /$depthai_ros_dir/install/setup.bash.."
source "/$depthai_ros_dir/install/setup.bash"
echo "ROS Distro: $ROS_DISTRO
source "$HOME/.bashrc"
exec "$@"
