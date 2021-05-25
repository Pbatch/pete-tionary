import boto3
from deep_daze import Imagine
import argparse


def run_deep_daze(prompt, seed, 
                  n_iterations=100,
                  num_layers=20,
                  lr=4e-5):
    model = Imagine(text=prompt.replace('_', ' '),
                    image_width=128,
                    num_layers=num_layers,
                    batch_size=64,
                    gradient_accumulate_every=1,
                    epochs=1,
                    seed=seed,
                    lr=lr,
                    save_progress=False,
                    iterations=n_iterations)
    model()


def create_image(prompt, bucket):
    save_path = f'./{prompt}.jpg'
    s3_client = boto3.client('s3')

    for seed in range(3):
        s3_path = f'prompt={prompt}-seed={seed}.jpg'
        run_deep_daze(prompt, seed)
        s3_client.upload_file(save_path, bucket, s3_path)


def main():
    parser = argparse.ArgumentParser(description='Run Deep Daze')
    parser.add_argument('-p', '--prompt', help='The text prompt', required=True)
    parser.add_argument('-b', '--bucket', help='The picture bucket', required=True)
    args = parser.parse_args()
    create_image(args.prompt, args.bucket)


if __name__ == '__main__':
    main()
