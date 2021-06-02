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

* Create our own version of Deep Daze
* Can shave off 12 seconds from Deep Daze by removing the clamp in `forward()`
* Can change the resolver for listMessages so that a filter can be added
* Can make the S3 URL checks a graphQL subscription (instead of pinging from Lambda)
* Create rooms

## Tables

* Message table, player and room table
    * Message - id, room, round, url, username
    * Room - id, name
* First person to create room becomes the admin. 
They start the game by deleting the room entry. 
This changes the mode of all the players to WRITE_PROMPT.

## Subscriptions

* Players have subscription to the room tale to know when the game is starting.
* Players have subscription to message table to know when the game is progressing.