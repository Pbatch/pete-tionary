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
    aws_batch as batch,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_ec2 as ec2
)


class PictionaryStack(core.Stack):
    SITE_SUB_DOMAIN = 'pictionary'
    DOMAIN_NAME = 'pbatch.net'
    SITE_DOMAIN = f'{SITE_SUB_DOMAIN}.{DOMAIN_NAME}'
    EC2_AMI = 'ami-082e298f790f88621'
    VPC_ID = 'vpc-0c367c61993de16df'
    PUBLIC_SUBNET_ID = 'subnet-0f65351474c1e561e'

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
            security_policy=cloudfront.SecurityPolicyProtocol.TLS_V1_1_2016
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
        asset_path = '../website/build'
        deployment = s3_deployment.BucketDeployment(self, 'my_deployment',
                                                    destination_bucket=self.website_bucket,
                                                    sources=[s3_deployment.Source.asset(asset_path)],
                                                    distribution=self.distribution,
                                                    distribution_paths=['/*'])
        return deployment

    def setup_picture_bucket(self):
        picture_bucket = s3.Bucket(self, 'my_picture_bucket',
                                   public_read_access=True,
                                   removal_policy=core.RemovalPolicy.DESTROY
                                   )
        return picture_bucket

    def setup_repository(self):
        repository_name = 'pictionary'
        repository = ecr.Repository.from_repository_name(self, 'my_repository',
                                                         repository_name=repository_name)
        return repository

    def setup_container(self):
        role = iam.Role(self, 'my_role', assumed_by=iam.ServicePrincipal('ecs-tasks.amazonaws.com'))
        policy = iam.PolicyStatement(actions=['s3:*'],
                                     resources=[self.picture_bucket.bucket_arn,
                                                f'{self.picture_bucket.bucket_arn}'])
        role.add_to_policy(policy)
        container = batch.JobDefinitionContainer(image=ecs.ContainerImage.from_ecr_repository(self.repository,
                                                                                              tag='latest'),
                                                 memory_limit_mib=15712,
                                                 job_role=role,
                                                 gpu_count=1,
                                                 vcpus=8)
        return container

    def setup_vpc(self):
        vpc = ec2.Vpc.from_vpc_attributes(self, 'my_vpc',
                                          vpc_id=self.VPC_ID,
                                          availability_zones=['eu-west-2a'],
                                          public_subnet_ids=[self.PUBLIC_SUBNET_ID])
        return vpc

    def setup_compute_environment(self):
        # There is no way to automatically get the latest ECS GPU machine image
        image = ec2.MachineImage.generic_linux({'eu-west-2': self.EC2_AMI})
        # A p2.2xlarge instance has 8 vcpus, 61 GiB memory and costs $3.06 per Hour
        instance_type = ec2.InstanceType('p3.2xlarge')
        compute_resources = batch.ComputeResources(vpc=self.vpc,
                                                   image=image,
                                                   instance_types=[instance_type],
                                                   maxv_cpus=8)
        compute_environment = batch.ComputeEnvironment(self, 'my_environment',
                                                       compute_resources=compute_resources)
        return compute_environment

    def setup_job_queue(self):
        job_queue_compute_environment = batch.JobQueueComputeEnvironment(
            compute_environment=self.compute_environment,
            order=1
        )
        job_queue = batch.JobQueue(self, 'my_job_queue',
                                   compute_environments=[job_queue_compute_environment])
        return job_queue

    def setup_job_definition(self):
        job_definition = batch.JobDefinition(self, 'my_job_definition',
                                             job_definition_name='pictionary',
                                             container=self.container)
        return job_definition

    def setup_lambda_function(self):
        lambda_function = lambda_.Function(self, 'my_lambda',
                                           handler='main.handler',
                                           runtime=lambda_.Runtime.PYTHON_3_7,
                                           code=lambda_.Code.from_asset('lambda'))
        policy = iam.PolicyStatement(actions=['batch:SubmitJob'],
                                     resources=["arn:aws:batch:*:*:*"])
        lambda_function.add_to_role_policy(policy)
        return lambda_function

    def setup_api(self):
        api = apigateway.LambdaRestApi(self, 'my_api',
                                       handler=self.lambda_function)
        return api

    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        self.website_bucket = self.setup_website_bucket()
        self.zone = self.setup_zone()
        self.distribution = self.setup_cloudfront_distribution()
        self.record = self.setup_route53_record()
        self.deployment = self.setup_bucket_deployment()
        self.picture_bucket = self.setup_picture_bucket()
        self.repository = self.setup_repository()
        self.container = self.setup_container()
        self.vpc = self.setup_vpc()
        self.compute_environment = self.setup_compute_environment()
        self.job_queue = self.setup_job_queue()
        self.job_definition = self.setup_job_definition()
        self.lambda_function = self.setup_lambda_function()
        self.api = self.setup_api()

        # Make CfnOutputs for variables we're interested in
        d = {'deploymentBucket': self.website_bucket.bucket_name,
             'pictureBucket': self.picture_bucket.bucket_name,
             'jobQueue': self.job_queue.job_queue_name,
             'jobDefinition': self.job_definition.job_definition_name,
             'websiteURL': f'https://{self.SITE_DOMAIN}',
             'apiEndpoint': self.api.url
             }
        for key, value in d.items():
            core.CfnOutput(self, key, value=value)
