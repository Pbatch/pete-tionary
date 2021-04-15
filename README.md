# Pictionary

## CDK

1) Make a private ECR repository called `pictionary`

2) `cd` into `docker`, then follow the commands in the `pictionary` repository to upload the Docker image

3) Set the global variables in `aws/pictionary/pictionary_stack.py`

4) Deploy the CloudFormation stack using CDK
```
cdk deploy --outputs-file "../website/src/constants/cloud.json"
```

## TODO

API Gateway times out after 29 seconds so we need to move the S3 query logic away from AWS Lambda to the client (in React).