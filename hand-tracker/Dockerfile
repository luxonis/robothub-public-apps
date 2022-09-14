FROM ros:galactic-ros-base-focal as base


RUN apt-get update && \
    apt-get install -q -y --no-install-recommends python3 python3-dev python3-pip python3-numpy ca-certificates ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-vision-msgs libboost-python-dev 

RUN sudo apt install -y \
  build-essential \
  cmake \
  git \
  python3-colcon-common-extensions \
  python3-pip \
  python3-rosdep \
  python3-setuptools \
  python

    # rm -rf /var/lib/apt/lists/*

WORKDIR /home
RUN mkdir -p dai_ws/src && cd dai_ws/src && git clone --depth=1 --branch hand-landmark  https://github.com/luxonis/depthai-ros.git
RUN pwd
RUN cd dai_ws && /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && pwd &&\
 colcon build --parallel-workers 3 --packages-select depthai_ros_msgs --event-handler console_direct+"

# RUN 
# Python will buffer output in case of non interactive terminals and we don't want that, because it delays logs
ENV PYTHONPATH=/lib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ARG DEPTHAI_BRANCH=main


FROM base as build
RUN apt-get update && apt-get install -q -y --no-install-recommends wget build-essential cmake pkg-config git
RUN wget https://github.com/libusb/libusb/releases/download/v1.0.25/libusb-1.0.25.tar.bz2 -O libusb.tar.bz2
RUN echo "--------- ${DEPTHAI_BRANCH}---------" 
# RUN git clone --depth=1 --recurse-submodules --branch main https://github.com/luxonis/depthai-python.git

RUN tar xf libusb.tar.bz2 \
    && cd libusb-* \
    && ./configure --disable-udev \
    && make -j$(nproc) \
    && make install

# RUN cd depthai-python \
#     && cmake -H. -B build -D CMAKE_BUILD_TYPE=Release -D DEPTHAI_ENABLE_BACKWARD=OFF \
#     && cmake --build build --parallel $(nproc)

# Package dependencies
# RUN mkdir -p /opt/depthai \
#     && for dep in $(ldd /depthai-python/build/depthai*.so 2>/dev/null | awk 'BEGIN{ORS=" "}$1 ~/^\//{print $1}$3~/^\//{print $3}' | sed 's/,$/\n/'); do cp "$dep" /opt/depthai; done \
#     && mv /depthai-py  thon/build/depthai*.so /opt/depthai

RUN pip3 install -U paho-mqtt paho_socket http_router opencv-python-headless depthai 

FROM base as stage

# COPY --from=build /opt/depthai /lib
# COPY --from=build /usr/local/lib/python3.8 /usr/local/lib/python3.8

# Clear Python compiled artifacts
# RUN find /usr -depth \
#     		\( \
#     			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
#     			-o \( -type f -a \( -name '*.pyc' -o -name '*.pyo' -o -name 'libpython*.a' \) \) \
#     		\) -exec rm -rf '{}' +

# Clear cache
RUN rm -rf /var/cache && mkdir -p /var/cache

FROM base

# Squash the image to save on space
COPY --from=stage / /

ADD ./docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ADD robothub-sdk/robothub_sdk/src/robothub_sdk /lib/robothub_sdk


ADD models models
# install packages
RUN apt update && apt install -q -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ADD FPS.py FPS.py
ADD HandTrackerBpfEdge.py HandTrackerBpfEdge.py
ADD HandTrackerRenderer.py HandTrackerRenderer.py
ADD mediapipe_utils.py mediapipe_utils.py

# Add as separate layer, since it changes often
ENTRYPOINT ["docker-entrypoint.sh"]
ARG FILE=app.py
ADD $FILE run.py

CMD ["python3", "/run.py"]