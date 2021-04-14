FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV PATH "$PATH:/root/.local/bin"

RUN apt-get update && \
    apt-get install gcc-8 g++-8 python3-pip -y && \
    pip3 install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install deep-daze boto3

CMD python3 main.py --prompt $prompt --bucket $bucket