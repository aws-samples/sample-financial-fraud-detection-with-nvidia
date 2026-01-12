import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export interface SageMakerInfraStackProps extends cdk.StackProps {
    dataBucketName: string;
    modelBucketName: string;
}

export class SageMakerInfraStack extends cdk.Stack {
    public readonly sagemakerExecutionRoleArn: string;
    public readonly sagemakerExecutionRole: iam.Role;
    public readonly pipelineExecutionRoleArn: string;
    public readonly pipelineExecutionRole: iam.Role;

    constructor(scope: Construct, id: string, props: SageMakerInfraStackProps) {
        super(scope, id, props);

        // Data Bucket (reference existing)
        const dataBucket = s3.Bucket.fromBucketName(
            this,
            "DataBucket",
            props.dataBucketName
        );

        // Model Bucket (reference existing)
        const modelBucket = s3.Bucket.fromBucketName(
            this,
            "ModelBucket",
            props.modelBucketName
        );

        // 1. SageMaker Execution Role
        // Used by Processing Jobs, Training Jobs, and Endpoints
        this.sagemakerExecutionRole = new iam.Role(this, "SageMakerExecutionRole", {
            assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description: "Execution role for SageMaker Jobs and Endpoints",
        });

        // Grant S3 access
        dataBucket.grantReadWrite(this.sagemakerExecutionRole);
        modelBucket.grantReadWrite(this.sagemakerExecutionRole);

        // Allow access to any S3 bucket starting with "sagemaker-" (common pattern)
        this.sagemakerExecutionRole.addToPolicy(new iam.PolicyStatement({
            actions: ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
            resources: ["arn:aws:s3:::sagemaker-*"],
        }));

        // Grant ECR access (for pulling custom containers)
        this.sagemakerExecutionRole.addManagedPolicy(
            iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonEC2ContainerRegistryReadOnly")
        );

        // Grant SageMaker Full Access
        this.sagemakerExecutionRole.addManagedPolicy(
            iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess")
        );

        // CloudWatch Logs
        this.sagemakerExecutionRole.addManagedPolicy(
            iam.ManagedPolicy.fromAwsManagedPolicyName("CloudWatchLogsFullAccess")
        );

        // Secrets Manager access (for NGC key if needed)
        this.sagemakerExecutionRole.addToPolicy(new iam.PolicyStatement({
            actions: ["secretsmanager:GetSecretValue"],
            resources: [`arn:aws:secretsmanager:${this.region}:${this.account}:secret:nvidia-ngc-api-key*`],
        }));

        this.sagemakerExecutionRoleArn = this.sagemakerExecutionRole.roleArn;

        // 2. Pipeline Execution Role
        // Orchestrates the pipeline steps
        this.pipelineExecutionRole = new iam.Role(this, "PipelineExecutionRole", {
            assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description: "Execution role for SageMaker Pipelines",
        });

        // Pipeline needs to pass role to jobs
        this.pipelineExecutionRole.addToPolicy(new iam.PolicyStatement({
            actions: ["iam:PassRole"],
            resources: [this.sagemakerExecutionRole.roleArn],
        }));

        // Pipeline needs sagemaker access
        this.pipelineExecutionRole.addManagedPolicy(
            iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess")
        );

        // S3 access for pipeline artifacts
        dataBucket.grantReadWrite(this.pipelineExecutionRole);
        modelBucket.grantReadWrite(this.pipelineExecutionRole);

        this.pipelineExecutionRoleArn = this.pipelineExecutionRole.roleArn;

        // Outputs
        new cdk.CfnOutput(this, "SageMakerExecutionRoleArn", {
            value: this.sagemakerExecutionRoleArn,
            exportName: "SageMakerExecutionRoleArn",
        });

        new cdk.CfnOutput(this, "PipelineExecutionRoleArn", {
            value: this.pipelineExecutionRoleArn,
            exportName: "PipelineExecutionRoleArn",
        });
    }
}
