# Pictionary

## Backend deployment with CDK

1.) Make a private ECR repository

2.) `cd` into `docker`, then follow the commands in the ECR repository to upload the Docker image

3.) Set the global variables in `aws/pictionary/pictionary_stack.py`

4.) Deploy the CloudFormation stack

```
cdk deploy --outputs-file "../frontend/src/constants/cdk.json"
```

## Future improvements

* Can shave off 12 seconds from Deep Daze by removing the clamp in `forward()`
* Can change the resolver for listMessages so that a filter can be added
* Can make the S3 URL checks a graphQL subscription (instead of pinging from Lambda)
* Set the hidden_size to 512

## Modes

0.) Wait for lobby

* How do rooms work?

1.) Wait for players

* Is there a room owner?
* Button disabled, input disabled, choice disabled

2.) Write prompt

* Button not disabled, input not disabled, choice disabled

3.) Wait for images

* Button disabled, input disabled, choice disabled

4.) Select image

* Button disabled, input disabled, choice enabled