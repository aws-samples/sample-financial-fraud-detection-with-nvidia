import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { Construct } from "constructs";

export interface TritonImageRepoStackProps extends cdk.StackProps {
  /**
   * GitHub repository URL for the source code
   * @default "https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia"
   */
  repoUrl?: string;

  /**
   * Branch to build from
   * @default "v2"
   */
  branch?: string;

  /**
   * Whether to trigger a build on stack deployment
   * @default true
   */
  triggerBuildOnDeploy?: boolean;
}

export class TritonImageRepoStack extends cdk.Stack {
  public readonly repository: ecr.Repository;
  public readonly repositoryUri: string;
  public readonly buildProject: codebuild.Project;

  constructor(scope: Construct, id: string, props?: TritonImageRepoStackProps) {
    super(scope, id, props);

    const repoUrl = props?.repoUrl ?? "https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia";
    const branch = props?.branch ?? "v2";
    const triggerBuildOnDeploy = props?.triggerBuildOnDeploy ?? true;

    // ECR Repository for custom Triton image
    this.repository = new ecr.Repository(this, "TritonInferenceRepo", {
      repositoryName: "triton-inference-server",
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
      lifecycleRules: [
        {
          maxImageCount: 5,
          description: "Keep only 5 most recent images",
        },
      ],
    });

    this.repositoryUri = this.repository.repositoryUri;

    // CodeBuild project - no source, clones repo in buildspec
    this.buildProject = new codebuild.Project(this, "TritonImageBuild", {
      projectName: "triton-inference-image-build",
      description: "Builds custom Triton image with PyTorch, PyG, XGBoost, Captum",
      buildSpec: codebuild.BuildSpec.fromObject({
        version: "0.2",
        phases: {
          pre_build: {
            commands: [
              "echo Logging in to Amazon ECR...",
              "aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com",
              "echo Cloning repository...",
              "git clone --depth 1 --branch $GIT_BRANCH $GIT_REPO_URL repo",
              "export COMMIT_HASH=$(cd repo && git rev-parse --short HEAD)",
              "echo Commit: $COMMIT_HASH",
            ],
          },
          build: {
            commands: [
              "echo Building Triton image...",
              "cd repo/triton",
              "docker build -t $ECR_REPO_URI:latest -t $ECR_REPO_URI:$COMMIT_HASH .",
            ],
          },
          post_build: {
            commands: [
              "echo Pushing to ECR...",
              "docker push $ECR_REPO_URI:latest",
              "docker push $ECR_REPO_URI:$COMMIT_HASH",
              "echo Build completed on `date`",
              "echo Image URI: $ECR_REPO_URI:latest",
            ],
          },
        },
      }),
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        privileged: true,
        computeType: codebuild.ComputeType.LARGE,
      },
      cache: codebuild.Cache.local(codebuild.LocalCacheMode.DOCKER_LAYER),
      environmentVariables: {
        ECR_REPO_URI: { value: this.repository.repositoryUri },
        AWS_ACCOUNT_ID: { value: this.account },
        AWS_REGION: { value: this.region },
        GIT_REPO_URL: { value: repoUrl },
        GIT_BRANCH: { value: branch },
        COMMIT_HASH: { value: "latest" },
      },
      timeout: cdk.Duration.hours(2),
      badge: false,
    });

    // Grant CodeBuild permission to push to ECR
    this.repository.grantPullPush(this.buildProject);

    // Grant CodeBuild permission to login to ECR
    this.buildProject.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["ecr:GetAuthorizationToken"],
        resources: ["*"],
      })
    );

    // Trigger build on deploy using Custom Resource
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
        codebuild = boto3.client('codebuild')
        project_name = event['ResourceProperties']['ProjectName']

        response = codebuild.start_build(projectName=project_name)
        build_id = response['build']['id']

        cfnresponse.send(event, context, cfnresponse.SUCCESS, {
            'BuildId': build_id
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        cfnresponse.send(event, context, cfnresponse.FAILED, {
            'Error': str(e)
        })
`),
      });

      triggerBuildFn.addToRolePolicy(
        new iam.PolicyStatement({
          actions: ["codebuild:StartBuild"],
          resources: [this.buildProject.projectArn],
        })
      );

      new cdk.CustomResource(this, "TriggerInitialBuild", {
        serviceToken: triggerBuildFn.functionArn,
        properties: {
          ProjectName: this.buildProject.projectName,
          // Change this to force a rebuild on redeploy
          BuildTrigger: Date.now().toString(),
        },
      });
    }

    // Outputs
    new cdk.CfnOutput(this, "TritonImageRepo", {
      value: this.repository.repositoryName,
      exportName: "TritonImageRepoName",
    });

    new cdk.CfnOutput(this, "TritonImageRepoUri", {
      value: this.repository.repositoryUri,
      exportName: "TritonImageRepoUri",
    });

    new cdk.CfnOutput(this, "TritonImageLatest", {
      value: `${this.repository.repositoryUri}:latest`,
      exportName: "TritonImageLatest",
    });

    new cdk.CfnOutput(this, "TritonBuildProject", {
      value: this.buildProject.projectName,
    });

    new cdk.CfnOutput(this, "TritonBuildProjectArn", {
      value: this.buildProject.projectArn,
    });
  }
}
