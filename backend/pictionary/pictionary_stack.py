from aws_cdk import (
    core,
    aws_s3 as s3,
    aws_s3_deployment as s3_deployment,
    aws_cloudfront as cloudfront,
    aws_certificatemanager as acm,
    aws_route53 as route53,
    aws_route53_targets as targets,
    aws_lambda as lambda_,
    aws_apigateway as apigateway,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_sqs as sqs,
    aws_cognito as cognito,
    aws_dynamodb as ddb,
    aws_appsync as appsync
)


class PictionaryStack(core.Stack):
    SITE_SUB_DOMAIN = 'pictionary'
    DOMAIN_NAME = 'pbatch.net'
    SITE_DOMAIN = f'{SITE_SUB_DOMAIN}.{DOMAIN_NAME}'
    # This has to be ECS-optimized GPU AMI
    # See https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-optimized_AMI.html
    EC2_AMI = 'ami-0510b115aae15fa15'
    VPC_ID = 'vpc-69eb0f10'
    REGION = 'eu-west-1'
    AVAILABILITY_ZONES = ['eu-west-1a',
                          'eu-west-1b',
                          'eu-west-1c']
    PUBLIC_SUBNET_IDS = ['subnet-f0ac3faa',
                         'subnet-27e1b741',
                         'subnet-4df5d405']
    INSTANCE_TYPE = 'g4dn.xlarge'
    VCPUS = 4
    MEMORY_LIMIT_MIB = 15500
    GPU_COUNT = 1

    def setup_queue(self):
        # Each job should take at most 2 minutes, so that is what the visibility timeout to be
        queue = sqs.Queue(self, 'my_queue',
                          visibility_timeout=core.Duration.minutes(2))
        return queue

    def setup_auth_lambda(self):
        auth_lambda = lambda_.Function(self, 'my_auth_lambda',
                                       handler='main.handler',
                                       runtime=lambda_.Runtime.PYTHON_3_7,
                                       code=lambda_.Code.from_asset('lambda/auth')
                                       )
        return auth_lambda

    def setup_user_pool(self):
        password_policy = cognito.PasswordPolicy(min_length=6,
                                                 require_digits=False,
                                                 require_lowercase=False,
                                                 require_symbols=False,
                                                 require_uppercase=False)
        user_pool_triggers = cognito.UserPoolTriggers(pre_sign_up=self.auth_lambda)
        user_pool = cognito.UserPool(self, 'my_user_pool',
                                     self_sign_up_enabled=True,
                                     password_policy=password_policy,
                                     lambda_triggers=user_pool_triggers,
                                     )
        return user_pool

    def setup_user_pool_client(self):
        user_pool_client = cognito.UserPoolClient(self, 'my_user_pool_client',
                                                  user_pool=self.user_pool)
        return user_pool_client

    def setup_message_table(self):
        partition_key = ddb.Attribute(name='id', type=ddb.AttributeType.STRING)
        message_table = ddb.Table(self, 'my_message_table',
                                  partition_key=partition_key,
                                  billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
                                  removal_policy=core.RemovalPolicy.DESTROY)
        return message_table

    def setup_room_table(self):
        partition_key = ddb.Attribute(name='id', type=ddb.AttributeType.STRING)
        room_table = ddb.Table(self, 'my_room_table',
                               partition_key=partition_key,
                               billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
                               removal_policy=core.RemovalPolicy.DESTROY)
        return room_table

    def setup_graphql_api(self):
        user_pool_config = appsync.UserPoolConfig(user_pool=self.user_pool,
                                                  default_action=appsync.UserPoolDefaultAction.ALLOW)
        default_authorization = appsync.AuthorizationMode(authorization_type=appsync.AuthorizationType.USER_POOL,
                                                          user_pool_config=user_pool_config)
        authorization_config = appsync.AuthorizationConfig(default_authorization=default_authorization)
        graphql_api = appsync.GraphqlApi(self, 'my_graphql_api',
                                         name='pictionary',
                                         schema=appsync.Schema.from_asset('graphql/schema.graphql'),
                                         authorization_config=authorization_config)
        return graphql_api

    def setup_message_table_data_source(self):
        message_table_data_source = self.graphql_api.add_dynamo_db_data_source(id='my_message_table_data_source',
                                                                               table=self.message_table)
        return message_table_data_source

    def setup_room_table_data_source(self):
        room_table_data_source = self.graphql_api.add_dynamo_db_data_source(id='my_room_table_data_source',
                                                                            table=self.room_table)
        return room_table_data_source

    def setup_create_message_resolver(self):
        key = appsync.PrimaryKey.partition('id').auto()
        values = appsync.Values.projecting('input')
        request_mapping_template = appsync.MappingTemplate.dynamo_db_put_item(key=key, values=values)
        response_mapping_template = appsync.MappingTemplate.dynamo_db_result_item()
        self.message_table_data_source.create_resolver(type_name='Mutation',
                                                       field_name='createMessage',
                                                       request_mapping_template=request_mapping_template,
                                                       response_mapping_template=response_mapping_template)

    def setup_list_messages_resolver(self):
        request_mapping_template = appsync.MappingTemplate.dynamo_db_scan_table()
        response_mapping_template = appsync.MappingTemplate.dynamo_db_result_item()
        self.message_table_data_source.create_resolver(type_name='Query',
                                                       field_name='listMessages',
                                                       request_mapping_template=request_mapping_template,
                                                       response_mapping_template=response_mapping_template)

    def setup_delete_message_resolver(self):
        request_mapping_template = appsync.MappingTemplate.dynamo_db_delete_item('id', 'id')
        response_mapping_template = appsync.MappingTemplate.dynamo_db_result_item()
        self.message_table_data_source.create_resolver(type_name='Mutation',
                                                       field_name='deleteMessage',
                                                       request_mapping_template=request_mapping_template,
                                                       response_mapping_template=response_mapping_template
                                                       )

    def setup_create_room_resolver(self):
        key = appsync.PrimaryKey.partition('id').auto()
        values = appsync.Values.projecting('input')
        request_mapping_template = appsync.MappingTemplate.dynamo_db_put_item(key=key, values=values)

        response_mapping_template = appsync.MappingTemplate.dynamo_db_result_item()
        self.room_table_data_source.create_resolver(type_name='Mutation',
                                                    field_name='createRoom',
                                                    request_mapping_template=request_mapping_template,
                                                    response_mapping_template=response_mapping_template)

    def setup_delete_room_resolver(self):
        request_mapping_template = appsync.MappingTemplate.dynamo_db_delete_item('id', 'id')
        response_mapping_template = appsync.MappingTemplate.dynamo_db_result_item()
        self.room_table_data_source.create_resolver(type_name='Mutation',
                                                    field_name='deleteRoom',
                                                    request_mapping_template=request_mapping_template,
                                                    response_mapping_template=response_mapping_template
                                                    )

    def setup_list_rooms_resolver(self):
        request_mapping_template = appsync.MappingTemplate.dynamo_db_scan_table()
        response_mapping_template = appsync.MappingTemplate.dynamo_db_result_item()
        self.room_table_data_source.create_resolver(type_name='Query',
                                                    field_name='listRooms',
                                                    request_mapping_template=request_mapping_template,
                                                    response_mapping_template=response_mapping_template)

    def setup_website_bucket(self):
        website_bucket = s3.Bucket(self, 'my_website_bucket',
                                   bucket_name=self.SITE_DOMAIN,
                                   public_read_access=True,
                                   removal_policy=core.RemovalPolicy.DESTROY,
                                   website_index_document='index.html',
                                   website_error_document='index.html')
        return website_bucket

    def setup_zone(self):
        zone = route53.HostedZone.from_lookup(self, 'my_zone',
                                              domain_name=self.DOMAIN_NAME)
        return zone

    def setup_cloudfront_distribution(self):
        certificate = acm.DnsValidatedCertificate(self, 'my_certificate',
                                                  domain_name=self.SITE_DOMAIN,
                                                  hosted_zone=self.zone,
                                                  region='us-east-1'  # CloudFront only checks this region for certs
                                                  )

        alias_configuration = cloudfront.AliasConfiguration(
            acm_cert_ref=certificate.certificate_arn,
            names=[self.SITE_DOMAIN],
            ssl_method=cloudfront.SSLMethod.SNI,
            security_policy=cloudfront.SecurityPolicyProtocol.TLS_V1_2_2019
        )
        custom_origin_source = cloudfront.CustomOriginConfig(
            domain_name=self.website_bucket.bucket_website_domain_name,
            origin_protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY
        )
        behaviors = [cloudfront.Behavior(is_default_behavior=True)]
        origin_configs = [cloudfront.SourceConfiguration(
            custom_origin_source=custom_origin_source,
            behaviors=behaviors
        )]
        distribution = cloudfront.CloudFrontWebDistribution(self, 'my_distribution',
                                                            alias_configuration=alias_configuration,
                                                            origin_configs=origin_configs
                                                            )
        return distribution

    def setup_route53_record(self):
        target = targets.CloudFrontTarget(distribution=self.distribution)
        record = route53.ARecord(self, 'my_record',
                                 record_name=self.SITE_DOMAIN,
                                 target=route53.RecordTarget.from_alias(target),
                                 zone=self.zone)
        return record

    def setup_bucket_deployment(self):
        asset_path = '../frontend/build'
        deployment = s3_deployment.BucketDeployment(self, 'my_deployment',
                                                    destination_bucket=self.website_bucket,
                                                    sources=[s3_deployment.Source.asset(asset_path)],
                                                    distribution=self.distribution,
                                                    distribution_paths=['/*'])
        return deployment

    def setup_picture_bucket(self):
        picture_bucket = s3.Bucket(self, 'my_picture_bucket',
                                   removal_policy=core.RemovalPolicy.DESTROY,
                                   public_read_access=True
                                   )
        return picture_bucket

    def setup_repository(self):
        repository_name = 'pictionary'
        repository = ecr.Repository.from_repository_name(self, 'my_repository',
                                                         repository_name=repository_name)
        return repository

    def setup_vpc(self):
        vpc = ec2.Vpc.from_vpc_attributes(self, 'my_vpc',
                                          vpc_id=self.VPC_ID,
                                          availability_zones=self.AVAILABILITY_ZONES,
                                          public_subnet_ids=self.PUBLIC_SUBNET_IDS)
        return vpc

    def setup_cluster(self):
        cluster = ecs.Cluster(self, 'my_cluster',
                              vpc=self.vpc)
        return cluster

    def setup_autoscaling_group(self):
        machine_image = ec2.MachineImage.generic_linux({self.REGION: self.EC2_AMI})
        instance_type = ec2.InstanceType(self.INSTANCE_TYPE)
        auto_scaling_group = self.cluster.add_capacity('my_capacity',
                                                       instance_type=instance_type,
                                                       machine_image=machine_image,
                                                       min_capacity=1,
                                                       max_capacity=1,
                                                       allow_all_outbound=True)
        return auto_scaling_group

    def setup_task(self):
        task = ecs.TaskDefinition(self, 'my_task',
                                  compatibility=ecs.Compatibility('EC2'))

        s3_policy = iam.PolicyStatement(actions=['s3:PutObject'],
                                        resources=[self.picture_bucket.bucket_arn,
                                                   f'{self.picture_bucket.bucket_arn}/*'])
        task.add_to_task_role_policy(s3_policy)

        sqs_policy = iam.PolicyStatement(actions=['sqs:ReceiveMessage', 'sqs:DeleteMessage'],
                                         resources=[self.queue.queue_arn])
        task.add_to_task_role_policy(sqs_policy)

        return task

    def setup_container(self):
        image = ecs.ContainerImage.from_ecr_repository(self.repository, tag='latest')
        logging = ecs.AwsLogDriver(stream_prefix='pictionary')
        container = ecs.ContainerDefinition(self, 'my_container',
                                            task_definition=self.task,
                                            image=image,
                                            logging=logging,
                                            memory_reservation_mib=self.MEMORY_LIMIT_MIB,
                                            gpu_count=self.GPU_COUNT)
        return container

    def setup_images_lambda(self):
        images_lambda = lambda_.Function(self, 'my_images_lambda',
                                         handler='main.handler',
                                         runtime=lambda_.Runtime.PYTHON_3_7,
                                         code=lambda_.Code.from_asset('lambda/images')
                                         )

        # Give the lambda permission to list objects in S3
        s3_policy = iam.PolicyStatement(actions=['s3:ListBucket'],
                                        resources=[self.picture_bucket.bucket_arn,
                                                   f'{self.picture_bucket.bucket_arn}/*'])
        images_lambda.add_to_role_policy(s3_policy)

        # Give the lambda permission to send jobs to the SQS queue
        sqs_policy = iam.PolicyStatement(actions=['sqs:SendMessage'],
                                         resources=[self.queue.queue_arn])
        images_lambda.add_to_role_policy(sqs_policy)

        return images_lambda

    def setup_lambda_api(self):
        lambda_api = apigateway.LambdaRestApi(self, 'my_lambda_api',
                                              handler=self.images_lambda)
        return lambda_api

    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        self.queue = self.setup_queue()
        self.auth_lambda = self.setup_auth_lambda()
        self.user_pool = self.setup_user_pool()
        self.user_pool_client = self.setup_user_pool_client()
        self.message_table = self.setup_message_table()
        self.room_table = self.setup_room_table()
        self.graphql_api = self.setup_graphql_api()
        self.message_table_data_source = self.setup_message_table_data_source()
        self.room_table_data_source = self.setup_room_table_data_source()
        self.setup_create_message_resolver()
        self.setup_list_messages_resolver()
        self.setup_delete_message_resolver()
        self.setup_create_room_resolver()
        self.setup_list_rooms_resolver()
        self.setup_delete_room_resolver()
        self.website_bucket = self.setup_website_bucket()
        self.zone = self.setup_zone()
        self.distribution = self.setup_cloudfront_distribution()
        self.record = self.setup_route53_record()
        self.deployment = self.setup_bucket_deployment()
        self.picture_bucket = self.setup_picture_bucket()
        self.repository = self.setup_repository()
        self.vpc = self.setup_vpc()
        self.cluster = self.setup_cluster()
        self.setup_autoscaling_group = self.setup_autoscaling_group()
        self.task = self.setup_task()
        self.container = self.setup_container()
        self.images_lambda = self.setup_images_lambda()
        self.lambda_api = self.setup_lambda_api()

        # Make CfnOutputs for variables we're interested in
        d = {'deploymentBucket': self.website_bucket.bucket_name,
             'pictureBucket': self.picture_bucket.bucket_name,
             'websiteUrl': f'https://{self.SITE_DOMAIN}',
             'lambdaUrl': self.lambda_api.url,
             'graphqlUrl': self.graphql_api.graphql_url,
             'userPoolId': self.user_pool.user_pool_id,
             'userPoolClientId': self.user_pool_client.user_pool_client_id,
             'region': self.region,
             'container': self.container.container_name,
             'taskDefinitionArn': self.task.task_definition_arn,
             'clusterArn': self.cluster.cluster_arn,
             'queueUrl': self.queue.queue_url,
             }
        for key, value in d.items():
            core.CfnOutput(self, key, value=value)
