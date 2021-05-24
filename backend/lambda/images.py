import boto3
from botocore.exceptions import ClientError
import json

batch_client = boto3.client('batch')
s3_client = boto3.client('s3')


def generate_images(body):
    if check_images_ready(body) == 'ready':
        return f'job for prompt "{body["prompt"]}" already completed'

    container_overrides = {
        'environment': [
            {
                'name': 'prompt',
                'value': body['prompt']
            },
            {
                'name': 'bucket',
                'value': body['bucket']
            }
        ]
    }
    job_name = body['prompt']
    batch_client.submit_job(jobName=job_name,
                            jobQueue=body['jobQueue'],
                            jobDefinition=body['jobDefinition'],
                            containerOverrides=container_overrides
                            )
    return f'submitted job for prompt "{body["prompt"]}"'


def check_images_ready(body):
    keys = [f'prompt={body["prompt"]}-seed={i}.jpg' for i in range(3)]
    for key in keys:
        try:
            s3_client.head_object(Bucket=body['bucket'], Key=key)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                return 'not ready'
            else:
                print(e)
                raise ValueError('Something dodgy has happened')
    return 'ready'


def handler(event, context):
    print(event)

    body = json.loads(event['body'])
    if body['function'] == 'generate_images':
        response_body = generate_images(body)
    elif body['function'] == 'check_images_ready':
        response_body = check_images_ready(body)
    else:
        raise ValueError(f'Function {body["function"]} does not exist')

    response = {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(response_body)
    }

    print(response)
    return response
