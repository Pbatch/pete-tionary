from imagine import Imagine
from deep_daze import DeepDaze
from utils import download_vit, load_perceptor
import os
import argparse
import boto3
import torch
import random
import time

s3_client = boto3.client('s3')


def create_images(prompt, perceptor, bucket):
    # Generate the image
    model = DeepDaze(perceptor, n_images=3)
    imagine = Imagine(prompt=prompt, model=model)
    imagine()

    # Save the images
    save_paths = [f'prompt={prompt.replace(" ", "_")}-seed={seed}.jpg'
                  for seed in range(3)]
    imagine.save(save_paths)
    for save_path in save_paths:
        s3_client.upload_file(save_path, bucket, save_path)


def poll_queue(queue_url, bucket, region):
    # Load the CLIP perceptor
    path = os.path.expanduser('~/.cache/clip/ViT-B-32.pt')
    perceptor = load_perceptor(path)

    # Continuously poll the queue
    queue = boto3.resource('sqs', region_name=region).Queue(queue_url)
    while True:
        # Process one user at a time
        response = queue.receive_messages(MaxNumberOfMessages=1)
        if response:
            message = response[0]
            prompt = message.body
            create_images(prompt, perceptor, bucket)
            message.delete()
        else:
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description='Run Deep Daze')
    parser.add_argument('-q', '--queueUrl', help='The URL of the queue', required=True)
    parser.add_argument('-b', '--bucket', help='The picture bucket', required=True)
    parser.add_argument('-r', '--region', help='The region of the SQS queue', required=True)
    args = parser.parse_args()
    poll_queue(args.queueUrl, args.bucket, args.region)


if __name__ == '__main__':
    main()
