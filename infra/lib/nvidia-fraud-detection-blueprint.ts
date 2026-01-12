import * as cdk from "aws-cdk-lib";
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
    public readonly dataBucket: s3.Bucket;
    public readonly modelBucket: s3.Bucket;
    public readonly modelRegistryBucket: s3.Bucket;

    constructor(
        scope: Construct,
        id: string,
        props: NvidiaFraudDetectionBlueprintProps,
    ) {
        super(scope, id, props);

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

        // Outputs
        new cdk.CfnOutput(this, "DataBucketName", {
            value: this.dataBucket.bucketName,
            exportName: "DataBucketName",
        });

        new cdk.CfnOutput(this, "ModelBucketName", {
            value: this.modelBucket.bucketName,
            exportName: "ModelBucketName",
        });
    }
}
