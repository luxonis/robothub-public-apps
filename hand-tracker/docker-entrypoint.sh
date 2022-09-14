#!/bin/bash

echo "Running docker-entrypoint $*"

source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/workspace/install/setup.bash"
source "/home/install/setup.bash"

exec "$@"
