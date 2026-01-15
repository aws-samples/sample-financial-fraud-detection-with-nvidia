import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { Construct } from "constructs";

export interface SageMakerPreprocessingImageRepoStackProps
  extends cdk.StackProps {
  repoUrl?: string;
  branch?: string;
  triggerBuildOnDeploy?: boolean;
}

export class SageMakerPreprocessingImageRepoStack extends cdk.Stack {
  public readonly repository: ecr.Repository;
  public readonly repositoryUri: string;
  public readonly buildProject: codebuild.Project;

  constructor(
    scope: Construct,
    id: string,
    props?: SageMakerPreprocessingImageRepoStackProps,
  ) {
    super(scope, id, props);

    const repoUrl =
      props?.repoUrl ??
      "https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia";
    const branch = props?.branch ?? "v2_sagemaker";
    const triggerBuildOnDeploy = props?.triggerBuildOnDeploy ?? true;

    // ECR Repository
    this.repository = new ecr.Repository(this, "SageMakerPreprocessingRepo", {
      repositoryName: "rapids-preprocessing",
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
      lifecycleRules: [{ maxImageCount: 5 }],
    });

    this.repositoryUri = this.repository.repositoryUri;

    // CodeBuild Project
    this.buildProject = new codebuild.Project(
      this,
      "SageMakerPreprocessingImageBuild",
      {
        projectName: "sagemaker-preprocessing-image-build",
        description: "Builds RAPIDS preprocessing image for SageMaker",
        buildSpec: codebuild.BuildSpec.fromObject({
          version: "0.2",
          phases: {
            pre_build: {
              commands: [
                "echo Logging in to NGC...",
                "export NGC_API_KEY=$(aws secretsmanager get-secret-value --secret-id nvidia-ngc-api-key --query SecretString --output text)",
                "echo $NGC_API_KEY | docker login nvcr.io -u '$oauthtoken' --password-stdin",
                "echo Logging in to Amazon ECR...",
                "aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com",
                "echo Cloning repository...",
                "git clone --depth 1 --branch $GIT_BRANCH $GIT_REPO_URL repo",
                "export COMMIT_HASH=$(cd repo && git rev-parse --short HEAD)",
              ],
            },
            build: {
              commands: [
                "echo Building Preprocessing image...",
                "cd repo",
                "docker build -f infra/docker/rapids-preprocessing/Dockerfile -t $ECR_REPO_URI:latest -t $ECR_REPO_URI:$COMMIT_HASH .",
              ],
            },
            post_build: {
              commands: [
                "echo Pushing to ECR...",
                "docker push $ECR_REPO_URI:latest",
                "docker push $ECR_REPO_URI:$COMMIT_HASH",
              ],
            },
          },
        }),
        environment: {
          buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
          privileged: true,
          computeType: codebuild.ComputeType.LARGE,
        },
        environmentVariables: {
          ECR_REPO_URI: { value: this.repository.repositoryUri },
          AWS_ACCOUNT_ID: { value: this.account },
          AWS_REGION: { value: this.region },
          GIT_REPO_URL: { value: repoUrl },
          GIT_BRANCH: { value: branch },
        },
        timeout: cdk.Duration.hours(1),
      },
    );

    this.repository.grantPullPush(this.buildProject);

    // Grant CodeBuild permission to read NGC API key from Secrets Manager
    this.buildProject.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["secretsmanager:GetSecretValue"],
        resources: [
          `arn:aws:secretsmanager:${this.region}:${this.account}:secret:nvidia-ngc-api-key*`,
        ],
      }),
    );

    // Custom Resource trigger (reusing the pattern)
    if (triggerBuildOnDeploy) {
      const triggerBuildFn = new lambda.Function(this, "TriggerBuildFunction", {
        runtime: lambda.Runtime.PYTHON_3_12,
        handler: "index.handler",
        timeout: cdk.Duration.minutes(1),
        code: lambda.Code.fromInline(`
import boto3
import cfnresponse
def handler(event, context):
    if event['RequestType'] == 'Delete':
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
        return
    try:
        cb = boto3.client('codebuild')
        cb.start_build(projectName=event['ResourceProperties']['ProjectName'])
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
    except Exception as e:
        print(str(e))
        cfnresponse.send(event, context, cfnresponse.FAILED, {})
`),
      });

      triggerBuildFn.addToRolePolicy(
        new iam.PolicyStatement({
          actions: ["codebuild:StartBuild"],
          resources: [this.buildProject.projectArn],
        }),
      );

      new cdk.CustomResource(this, "TriggerBuild", {
        serviceToken: triggerBuildFn.functionArn,
        properties: {
          ProjectName: this.buildProject.projectName,
          BuildTrigger: Date.now().toString(),
        },
      });
    }

    new cdk.CfnOutput(this, "PreprocessingRepoUri", {
      value: this.repositoryUri,
    });
  }
}
