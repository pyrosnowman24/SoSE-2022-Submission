FROM ubuntu:20.04

# Install libraries needed for imports
RUN apt-get upgrade && \
    apt-get update && \
    apt-get install -y  \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
RUN pip3 install --upgrade pip3
WORKDIR \app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/pyrosnowman24/Map_Dataset_Generator.git
