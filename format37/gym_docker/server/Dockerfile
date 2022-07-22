# A Dockerfile that sets up a full Gym install with test dependencies
# set PYTHON_VERSION to 3.6.15
ARG PYTHON_VERSION=3.6.15
#FROM python:$PYTHON_VERSION
#FROM nvidia/cuda:11.7.0-devel-ubuntu20.04
#FROM nvidia/cuda:11.7.0-runtime-ubuntu20.04 # Build ok, but too new for mi driver
FROM nvidia/cuda:10.2-runtime-ubuntu18.04

#SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    unzip \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf \
    ffmpeg cmake \
    swig \
    && apt-get install -y wget \
    && apt-get install -y python3-pip \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Download mujoco
    && mkdir /root/.mujoco \
    && cd /root/.mujoco \
    && wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin"

COPY . /usr/local/gym/
WORKDIR /usr/local/gym/
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
COPY server.py .
#RUN if [ "python:${PYTHON_VERSION}" = "python:3.6.15" ] ; then pip install .[box2d,classic_control,toy_text,other] pytest=="7.0.1" --no-cache-dir; else pip install .[testing] --no-cache-dir; fi
#RUN pip install .[testing] --no-cache-dir

#ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
ENTRYPOINT ["python3"]
CMD ["server.py"]