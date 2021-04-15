import boto3
from botocore.errorfactory import ClientError
import json
import time

batch_client = boto3.client('batch')
s3_client = boto3.client('s3')


def keys_in_bucket(keys, bucket):
    for key in keys:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
        except ClientError:
            return False
    return True


def submit_batch_job(body):
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


def handler(event, context):
    print(event)
    body = json.loads(event['body'])

    prefix = f'prompt={body["prompt"]}'
    keys = [f'{prefix}-seed={i}.jpg' for i in range(3)]
    if not keys_in_bucket(keys, body['bucket']):
        submit_batch_job(body)
        while True:
            if keys_in_bucket(keys, body['bucket']):
                break
            time.sleep(1)

    location = s3_client.get_bucket_location(Bucket=body['bucket'])['LocationConstraint']
    output = json.dumps({'urls': [f'https://s3-{location}.amazonaws.com/{body["bucket"]}/{key}'
                                  for key in keys]})

    response = {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
        'body': output
    }

    return response
