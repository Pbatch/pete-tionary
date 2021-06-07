import boto3
import json


def main():
    client = boto3.client('ecs')
    with open('frontend/src/constants/cdk.json', 'r') as f:
        CDK_CONFIG = json.loads(f.read())['pictionary']

    overrides = {
        'containerOverrides': [
            {
                'name': CDK_CONFIG['container'],
                'environment': [
                    {
                        'name': 'bucket',
                        'value': CDK_CONFIG['pictureBucket']
                    },
                    {
                        'name': 'queueUrl',
                        'value': CDK_CONFIG['queueUrl']
                    },
                    {
                        'name': 'region',
                        'value': CDK_CONFIG['region']
                    }
                ]
            }
        ]
    }
    response = client.run_task(
        cluster=CDK_CONFIG['clusterArn'],
        taskDefinition=CDK_CONFIG['taskDefinitionArn'],
        overrides=overrides
    )
    print(response)


if __name__ == '__main__':
    main()
