FROM ghcr.io/luxonis/robothub-base-app:ubuntu-depthai-main as base

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*


# setup sources.list and keys for Galactic
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# RUN echo "deb http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO galactic

# install ros2 packages and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ros-base \
    libboost-python-dev \
    ros-${ROS_DISTRO}-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U opencv-python-headless

ADD docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ADD models models

# install packages
RUN apt update && apt install -q -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -q -y --no-install-recommends \
    python3-dev \
    gcc \
    libc-dev \
    iputils-ping \
    git \
    python3-opencv

WORKDIR /home
RUN mkdir -p dai_ws/src && RUN git clone https://github.com/luxonis/depthai-ros.git --branch hand-landmark && \
        cd .. && source "/opt/ros/$ROS_DISTRO/setup.bash" && colcon build --packages-select depthai_ros_msgs



ADD FPS.py FPS.py
ADD HandTrackerBpfEdge.py HandTrackerBpfEdge.py
ADD HandTrackerRenderer.py HandTrackerRenderer.py
ADD mediapipe_utils.py mediapipe_utils.py

ARG FILE=app.py
ADD $FILE run.py