import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { Construct } from "constructs";

export interface SageMakerTrainingImageRepoStackProps extends cdk.StackProps {
    /**
     * GitHub repository URL for the source code
     * @default "https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia"
     */
    repoUrl?: string;

    /**
     * Branch to build from
     * @default "v2_sagemaker"
     */
    branch?: string;

    /**
     * Whether to trigger a build on stack deployment
     * @default true
     */
    triggerBuildOnDeploy?: boolean;
}

export class SageMakerTrainingImageRepoStack extends cdk.Stack {
    public readonly repository: ecr.Repository;
    public readonly repositoryUri: string;
    public readonly buildProject: codebuild.Project;

    constructor(scope: Construct, id: string, props?: SageMakerTrainingImageRepoStackProps) {
        super(scope, id, props);

        const repoUrl = props?.repoUrl ?? "https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia";
        // Default to v2_sagemaker if not specified, or fallback to v2/main if needed. 
        // The plan says "v2_sagemaker" branch.
        const branch = props?.branch ?? "v2_sagemaker";
        const triggerBuildOnDeploy = props?.triggerBuildOnDeploy ?? true;

        // ECR Repository for SageMaker training image
        this.repository = new ecr.Repository(this, "SageMakerTrainingRepo", {
            repositoryName: "nvidia-training-repo-sagemaker",
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

        // CodeBuild project - pulls NGC image and pushes to ECR
        this.buildProject = new codebuild.Project(this, "SageMakerTrainingImageBuild", {
            projectName: "sagemaker-training-image-copy",
            description: "Copies NGC training image to ECR for SageMaker",
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
                            "echo Commit: $COMMIT_HASH",
                        ],
                    },
                    build: {
                        commands: [
                            "echo Building SageMaker Training image...",
                            "cd repo",
                            "docker build -f infra/docker/training-sagemaker/Dockerfile -t $ECR_REPO_URI:latest -t $ECR_REPO_URI:$COMMIT_HASH .",
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
            environmentVariables: {
                ECR_REPO_URI: { value: this.repository.repositoryUri },
                AWS_ACCOUNT_ID: { value: this.account },
                AWS_REGION: { value: this.region },
                GIT_REPO_URL: { value: repoUrl },
                GIT_BRANCH: { value: branch },
            },
            timeout: cdk.Duration.hours(1),
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

        // Grant CodeBuild permission to read NGC API key from Secrets Manager
        this.buildProject.addToRolePolicy(
            new iam.PolicyStatement({
                actions: ["secretsmanager:GetSecretValue"],
                resources: [`arn:aws:secretsmanager:${this.region}:${this.account}:secret:nvidia-ngc-api-key*`],
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
        new cdk.CfnOutput(this, "SageMakerTrainingRepoName", {
            value: this.repository.repositoryName,
            exportName: "SageMakerTrainingRepoName",
        });

        new cdk.CfnOutput(this, "SageMakerTrainingRepoUri", {
            value: this.repository.repositoryUri,
            exportName: "SageMakerTrainingRepoUri",
        });

        new cdk.CfnOutput(this, "SageMakerTrainingImageLatest", {
            value: `${this.repository.repositoryUri}:latest`,
            exportName: "SageMakerTrainingImageLatest",
        });
    }
}
