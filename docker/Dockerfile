FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

ENV PATH "$PATH:/root/.local/bin"

RUN apt-get update && \
    apt-get install gcc-8 g++-8 python3-pip libgl1-mesa-glx -y && \
    pip3 install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install ftfy boto3 regex basicsr packaging

# Copy the files required for downloading the CLIP weights
COPY clip.py utils.py download.py ./

# Download the CLIP weights
RUN python3 download.py

# Copy everything else
COPY main.py direct_visions.py tokenizer.py bpe_simple_vocab_16e6.txt x4plus.pth ./

CMD python3 main.py --bucket $bucket --queueUrl $queueUrl --region $region