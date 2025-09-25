import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export interface SageMakerExecutionRoleStackProps extends cdk.StackProps {
  /**
   * The S3 bucket ARN where the models, data and config is stored
   */
  modelBucketArn: string;
}

export class SageMakerExecutionRoleStack extends cdk.Stack {
  public readonly sagemakerRole: iam.Role;

  constructor(
    scope: Construct,
    id: string,
    props?: SageMakerExecutionRoleStackProps,
  ) {
    super(scope, id, props);

    // Create the SageMaker execution role
    this.sagemakerRole = new iam.Role(this, "SageMakerExecutionRole", {
      roleName: "AmazonSageMaker-ExecutionRole-CDK",
      description:
        "SageMaker execution role created with CDK matching existing permissions",
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
      maxSessionDuration: cdk.Duration.hours(1),

      // AWS Managed Policies
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess"),
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AmazonSageMakerCanvasFullAccess",
        ),
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AmazonSageMakerCanvasAIServicesAccess",
        ),
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AmazonSageMakerCanvasDataPrepFullAccess",
        ),
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AmazonSageMakerCanvasSMDataScienceAssistantAccess",
        ),
      ],
    });

    // Custom inline policy for S3 SageMaker bucket access
    const sagemakerS3Policy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["s3:ListBucket"],
          resources: ["arn:aws:s3:::SageMaker"],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
          resources: ["arn:aws:s3:::SageMaker/*", props!.modelBucketArn],
        }),
      ],
    });

    // Custom inline policy for SSM Parameter Store
    const parameterStorePolicy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "VisualEditor0",
          effect: iam.Effect.ALLOW,
          actions: ["ssm:PutParameter"],
          resources: [
            `arn:aws:ssm:${this.region}:${this.account}:parameter/triton/*`,
          ],
        }),
      ],
    });

    // Custom inline policy for SageMaker StartSession
    const sagemakerStartSessionPolicy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "Statement1",
          effect: iam.Effect.ALLOW,
          actions: ["sagemaker:StartSession"],
          resources: ["*"],
        }),
      ],
    });

    // Attach inline policies to the role
    this.sagemakerRole.attachInlinePolicy(
      new iam.Policy(this, "SageMakerS3Policy", {
        policyName: "SageMaker-S3-Access",
        document: sagemakerS3Policy,
      }),
    );

    this.sagemakerRole.attachInlinePolicy(
      new iam.Policy(this, "ParameterStorePolicy", {
        policyName: "parameter-store",
        document: parameterStorePolicy,
      }),
    );

    this.sagemakerRole.attachInlinePolicy(
      new iam.Policy(this, "SageMakerStartSessionPolicy", {
        policyName: "sagemaker-start-session",
        document: sagemakerStartSessionPolicy,
      }),
    );

    // Output the role ARN
    new cdk.CfnOutput(this, "SageMakerRoleArn", {
      value: this.sagemakerRole.roleArn,
      description: "ARN of the SageMaker execution role",
      exportName: "SageMakerExecutionRoleArn",
    });
  }
}
