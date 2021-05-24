FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

ENV PATH "$PATH:/root/.local/bin"

RUN apt-get update && \
    apt-get install gcc-8 g++-8 python3-pip -y && \
    pip3 install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install deep-daze boto3

COPY main.py download.py ./
    
RUN python3 download.py

CMD python3 main.py --prompt $prompt --bucket $bucket