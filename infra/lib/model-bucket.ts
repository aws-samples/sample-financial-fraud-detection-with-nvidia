import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';

export interface S3BucketStackProps extends cdk.StackProps {
    bucketName: string;
}

export class S3BucketStack extends cdk.Stack {
        constructor(scope: cdk.App, id: string, props?: S3BucketStackProps) {
        super(scope, id, props);

        const bucket = new s3.Bucket(this, 'NvidiaFraudDetectionBlueprintBucket', {
            bucketName: props?.bucketName,
            blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
            encryption: s3.BucketEncryption.S3_MANAGED,
        });

        new cdk.CfnOutput(this, 'BucketName', {
            value: bucket.bucketName
        });

        new cdk.CfnOutput(this, 'BucketArn', {
            value: bucket.bucketArn
        });
    }
}