import boto3
import json

batch_client = boto3.client('batch')


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

    submit_batch_job(body)

    response = {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
    }

    return response
