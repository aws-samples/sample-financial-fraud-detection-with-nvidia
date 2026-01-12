import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export interface NvidiaFraudDetectionBlueprintProps extends cdk.StackProps {
    /**
     * The S3 bucket name where models are stored
     */
    modelBucketName: string;

    /**
     * The S3 bucket name for raw data (e.g., TabFormer dataset)
     */
    dataBucketName: string;

    /**
     * The S3 bucket name for model registry (Triton model repository)
     */
    modelRegistryBucketName: string;
}

export class NvidiaFraudDetectionBlueprint extends cdk.Stack {
    public readonly vpc: ec2.Vpc;
    public readonly dataBucket: s3.Bucket;
    public readonly modelBucket: s3.Bucket;
    public readonly modelRegistryBucket: s3.Bucket;

    constructor(
        scope: Construct,
        id: string,
        props: NvidiaFraudDetectionBlueprintProps,
    ) {
        super(scope, id, props);

        // Use existing VPC or create a new one
        this.vpc = new ec2.Vpc(this, "TritonVpc", {
            maxAzs: 3,
            natGateways: 1,
            enableDnsHostnames: true,
            enableDnsSupport: true,
            flowLogs: {
                VpcFlowLog: {
                    destination: ec2.FlowLogDestination.toCloudWatchLogs(),
                },
            },
        });

        // Data Bucket
        this.dataBucket = new s3.Bucket(this, "DataBucket", {
            bucketName: props.dataBucketName,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
            serverAccessLogsPrefix: "access-logs/",
            enforceSSL: true,
            encryption: s3.BucketEncryption.S3_MANAGED,
        });

        // Model Bucket
        this.modelBucket = new s3.Bucket(this, "ModelBucket", {
            bucketName: props.modelBucketName,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
            serverAccessLogsPrefix: "access-logs/",
            enforceSSL: true,
            encryption: s3.BucketEncryption.S3_MANAGED,
        });

        // Model Registry Bucket (if different, otherwise we can reuse modelBucket)
        if (props.modelRegistryBucketName !== props.modelBucketName) {
            this.modelRegistryBucket = new s3.Bucket(this, "ModelRegistryBucket", {
                bucketName: props.modelRegistryBucketName,
                removalPolicy: cdk.RemovalPolicy.DESTROY,
                autoDeleteObjects: true,
                serverAccessLogsPrefix: "access-logs/",
                enforceSSL: true,
                encryption: s3.BucketEncryption.S3_MANAGED,
            });
        } else {
            this.modelRegistryBucket = this.modelBucket;
        }

        // VPC Interface Endpoints for SageMaker (Optional but recommended for security/cost)
        this.vpc.addInterfaceEndpoint("SageMakerRuntimeEndpoint", {
            service: ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_RUNTIME,
        });

        this.vpc.addInterfaceEndpoint("SageMakerApiEndpoint", {
            service: ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_API,
        });

        this.vpc.addGatewayEndpoint("S3Endpoint", {
            service: ec2.GatewayVpcEndpointAwsService.S3,
        });
    }
}
