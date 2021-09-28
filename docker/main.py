from direct_visions import DirectVisions
from utils import download_vit, load_perceptor, load_enhancer
import os
import argparse
import boto3
import time
import torch

s3_client = boto3.client('s3')


def create_images(prompt, perceptor, enhancer, bucket):
    # Generate the image
    model = DirectVisions(prompt=prompt,
                          perceptor=perceptor,
                          enhancer=enhancer,
                          n_images=3)
    model.run()

    # Save the images to S3
    for i in range(model.n_images):
        path = f'prompt={model.prompt.replace(" ", "_")}-seed={i}.jpg'
        s3_client.upload_file(path, bucket, path)
        os.remove(path)


def poll_queue(queue_url, bucket, region):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the CLIP perceptor
    perceptor_path = os.path.expanduser('~/.cache/clip/ViT-B-32.pt')
    perceptor = load_perceptor(perceptor_path).to(device)

    # Load the enhancer
    enhancer_path = 'x4plus.pth'
    enhancer = load_enhancer(enhancer_path).to(device)

    # Continuously poll the queue
    queue = boto3.resource('sqs', region_name=region).Queue(queue_url)
    while True:
        # Process one user at a time
        response = queue.receive_messages(MaxNumberOfMessages=1)
        if response:
            message = response[0]
            prompt = message.body
            create_images(prompt, perceptor, enhancer, bucket)
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
