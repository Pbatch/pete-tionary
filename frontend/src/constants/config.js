import _CDK_CONFIG from './cdk.json'

const CDK_CONFIG = _CDK_CONFIG["pictionary"]

export const AMPLIFY_CONFIG = {
  Auth: {
    region: CDK_CONFIG["region"],
    userPoolId: CDK_CONFIG["userPoolId"],
    userPoolWebClientId: CDK_CONFIG["userPoolClientId"]
  },
  aws_appsync_graphqlEndpoint: CDK_CONFIG["graphqlUrl"],
  aws_appsync_region: CDK_CONFIG["region"],
  aws_appsync_authenticationType: "AMAZON_COGNITO_USER_POOLS"
}

export const LAMBDA_CONFIG = {
  region: CDK_CONFIG["region"],
  queueUrl: CDK_CONFIG["queueUrl"],
  bucket: CDK_CONFIG["pictureBucket"],
  lambdaUrl: CDK_CONFIG["lambdaUrl"],
  bucketUrl: `https://s3-${CDK_CONFIG["region"]}.amazonaws.com/${CDK_CONFIG["pictureBucket"]}`
}
