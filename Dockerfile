FROM ros-humble-base-l4t-pytorch-onnx:r35.4.1 AS developer

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
   && apt-get -y install --no-install-recommends software-properties-common git libusb-1.0-0-dev wget python3-colcon-common-extensions

# Build and install depthai core library
WORKDIR /opt
RUN git clone --recurse-submodules https://github.com/luxonis/depthai-core.git &&\
 cd depthai-core && mkdir build && cd build && \
 cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON && make && make install

# Build and install depthai-ros packages
RUN git clone --branch $ROS_DISTRO https://github.com/luxonis/depthai-ros.git && cd depthai-ros && \
git clone https://github.com/ros/diagnostics.git &&\ 
. /opt/ros/$ROS_DISTRO/install/setup.bash && \
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
--cmake-args -DBUILD_TESTING=OFF \
--cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
--cmake-args -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
--cmake-args -DBUILD_SHARED_LIBS=ON \
--cmake-args -Ddepthai_DIR=/opt/depthai-core/build/install/lib/cmake/depthai
ENV depthai_ros_dir=/opt/depthai-ros/

# Create user account
ARG USER=ros
ARG UID=1000
ARG PASS=pass
RUN apt update && apt install -y ssh zsh && sed -i 's/^#X11UseLocalhost yes/X11UseLocalhost no/' /etc/ssh/sshd_config && useradd -u $UID -ms /bin/bash ${USER} && \
    echo "$USER:$PASS" | chpasswd && usermod -aG sudo ${USER}
USER ${USER}
WORKDIR /home/${USER}

# Setup ros installation / sourcing
RUN sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
RUN echo "if [ -f ${depthai_ros_dir}/install/setup.zsh ]; then source ${depthai_ros_dir}/install/setup.zsh; fi" >> $HOME/.zshrc
RUN echo 'eval "$(register-python-argcomplete3 ros2)"' >> $HOME/.zshrc
RUN echo 'eval "$(register-python-argcomplete3 colcon)"' >> $HOME/.zshrc
RUN echo "if [ -f ${depthai_ros_dir}/install/setup.bash ]; then source ${depthai_ros_dir}/install/setup.bash; fi" >> $HOME/.bashrc


FROM developer as runtime
#TODO: install python node
#TODO: setup launch script

# Create an entrypoint that launches the node
ADD entrypoint.sh /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]
CMD ["ros2","run","depthai_examples","stereo_node"]


