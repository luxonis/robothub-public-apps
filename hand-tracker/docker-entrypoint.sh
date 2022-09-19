#!/bin/bash

echo "Running docker-entrypoint $*"

source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/home/dai_ws/install/setup.bash"

exec "$@"
