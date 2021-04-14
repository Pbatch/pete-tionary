import boto3
import json
batch_client = boto3.client('batch')


def handler(event, context):
    print(event)

    body = json.loads(event['body'])
    print(body)

    print('Submitting a job to the cluster')
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

    print('Getting urls')
    PLACEHOLDER_URL = 'https://pictionary-mypicturebucket4cd3a363-o8clxqyh5mhx.s3.eu-west-2.amazonaws.com/placeholder.png'
    output = json.dumps({'urls': [PLACEHOLDER_URL for _ in range(3)]})

    print('Sending a response')
    response = {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
        'body': output
    }
    print(response)

    return response
