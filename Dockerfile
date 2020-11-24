FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV APP_ROOT /app

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
# to support install openjdk-11-jre-headless
RUN mkdir -p /usr/share/man/man1
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    build-essential \
    git \
    bzip2 \
    ca-certificates \
    libssl-dev \
    libmysqlclient-dev \
    default-libmysqlclient-dev \
    make \
    cmake \
    protobuf-compiler \
    curl \
    sudo \
    software-properties-common \
    xz-utils \
    file \
    mecab \
    libmecab-dev \
    #   mecab-ipadic-utf8 \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-venv \
    openjdk-11-jre-headless \
    && curl -sL https://deb.nodesource.com/setup_10.x | bash - \
    && apt-get update && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc
RUN pip3 install -U pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt \
    pip install hydra-core --upgrade
RUN echo "すもももももももものうち中居正広"|mecab

RUN pipdeptree

# RUN mkdir data
# COPY data/*.jsonl data/
COPY app.py ./

CMD ["python3", "app.py"]