FROM ghcr.io/luxonis/robothub-app:2022.274.1807-ros2noetic-custom-dev

# setup timezone

# ENV ROS_DISTRO galactic
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends ca-certificates ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-vision-msgs libboost-python-dev 

RUN apt-get install -y \
  build-essential \
  cmake \
  git 
#   python3-colcon-common-extensions \
#   python3-pip \
#   python3-rosdep \
#   python3-setuptools \
#   python

    # rm -rf /var/lib/apt/lists/*

WORKDIR /home
RUN mkdir -p dai_ws/src && cd dai_ws/src && git clone --depth=1 --branch hand-landmark  https://github.com/luxonis/depthai-ros.git
RUN pwd
RUN cd dai_ws && /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash &&\ 
                    catkin_make -DCMAKE_INSTALL_PREFIX=/opt/ros/${ROS_DISTRO} install --only-pkg-with-deps depthai_ros_msgs"


ADD ./docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ADD robothub-sdk/robothub_sdk/src/robothub_sdk /lib/robothub_sdk

ADD models models
ADD custom_models custom_models

# install packages
# RUN apt update && apt install -q -y --no-install-recommends \
#     python3-pip \
#     && rm -rf /var/lib/apt/lists/*

RUN rm -rf /var/lib/apt/lists/*
# RUN apt update && apt install -q -y --no-install-recommends \
#     python3-dev \
#     gcc \
#     libc-dev \
#     iputils-ping 

ADD requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

ADD FPS.py FPS.py
ADD HandTrackerBpfEdge.py HandTrackerBpfEdge.py
ADD HandTrackerRenderer.py HandTrackerRenderer.py
ADD mediapipe_utils.py mediapipe_utils.py
ADD template_manager_script_bpf_solo.py template_manager_script_bpf_solo.py
ADD template_manager_script_bpf_duo.py template_manager_script_bpf_duo.py 

# Ad as separate layer, since it changes often
# ENTRYPOINT ["docker-entrypoint.sh"]
ARG FILE=app.py
ADD $FILE run.py    

# CMD ["python3", "run.py"]