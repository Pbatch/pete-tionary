import boto3
from deep_daze import Imagine

s3_client = boto3.client('s3')

def run_deep_daze(prompt, n_iterations=100):
    model = Imagine(text=prompt,
                image_width=128,
                num_layers=42,
                batch_size=64,
                gradient_accumulate_every=1,
                epochs=1,
                seed=seed,
                save_every=N_ITERATIONS,
                iterations=N_ITERATIONS)
    model()

def run_seed_writer(prompt, seed, save_path):
    with open(save_path, 'w') as f:
        f.write(seed)

def create_image(prompt, bucket):    
    no_space_prompt = prompt.replace(" ","_")
    save_path = f'./{no_space_prompt}.jpg'
    for seed in range(3):
        run_seed_writer(prompt, seed, save_path)
        s3_path = f'prompt={no_space_prompt}-seed={seed}.jpg'
        s3_client.upload_file(save_path, bucket, s3_path)

def create_text():
    s3_client.

def main():
    parser = argparse.ArgumentParser(description='Run Deep Daze')
    parser.add_argument('-p', '--prompt', help='The text prompt', required=True)
    parser.add_argument('-b', '--bucket', help='The picture bucket', required=True)
    args = parser.parse_args()
    create_image(args.prompt, args.bucket)

if __name__ == '__main__':
    main()