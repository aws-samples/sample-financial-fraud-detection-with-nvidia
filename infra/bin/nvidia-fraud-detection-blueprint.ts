import * as cdk from "aws-cdk-lib";
import { NvidiaFraudDetectionBlueprint } from "../lib/nvidia-fraud-detection-blueprint";
import { SageMakerTrainingImageRepoStack } from "../lib/sagemaker-training-image-repo";
import { SageMakerPreprocessingImageRepoStack } from "../lib/sagemaker-preprocessing-image-repo";
import { TritonImageRepoStack } from "../lib/triton-image-repo";
import { NeptuneGraphStack } from "../lib/neptune-graph-stack";
import { SageMakerInfraStack } from "../lib/sagemaker-infrastructure-stack";
import { SageMakerTritonEndpointStack } from "../lib/sagemaker-triton-endpoint-stack";
import { SageMakerDomainStack } from "../lib/sagemaker-domain-stack";
import { VpcPeeringStack } from "../lib/vpc-peering-stack";

const app = new cdk.App();

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION,
};

const repoUrl =
  "https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia.git";
const branch = "feature/neptune-graph-backend";

// Config
const modelBucketName =
  "fraud-detection-" + process.env.CDK_DEFAULT_ACCOUNT + "-sm";
const dataBucketName = modelBucketName;
const modelRegistryBucketName = modelBucketName + "-model-registry";

// 1. Training Image Repo (SageMaker)
const trainingImageRepo = new SageMakerTrainingImageRepoStack(
  app,
  "SageMakerTrainingImageRepoStack",
  {
    env: env,
    repoUrl: repoUrl,
    branch: branch,
  },
);

const preprocessingImageRepo = new SageMakerPreprocessingImageRepoStack(
  app,
  "SageMakerPreprocessingImageRepoStack",
  {
    env: env,
    repoUrl: repoUrl,
    branch: branch,
  },
);

// 2. Inference Image Repo (Triton)
const tritonImageRepo = new TritonImageRepoStack(app, "TritonImageRepoStack", {
  env: env,
  repoUrl: repoUrl,
  branch: branch,
});

// 3. Base Infrastructure (S3 buckets)
const baseInfra = new NvidiaFraudDetectionBlueprint(
  app,
  "NvidiaFraudDetectionBlueprint",
  {
    env: env,
    modelBucketName: modelBucketName,
    dataBucketName: dataBucketName,
    modelRegistryBucketName: modelRegistryBucketName,
  },
);

// 4. SageMaker IAM Roles & Infrastructure
const smInfra = new SageMakerInfraStack(app, "SageMakerInfraStack", {
  env: env,
  dataBucketName: dataBucketName,
  modelBucketName: modelBucketName,
});
smInfra.addDependency(baseInfra);

// 5a. Neptune Graph Database
const neptuneStack = new NeptuneGraphStack(app, "NeptuneGraphStack", {
  env: env,
  sagemakerExecutionRoleArn: smInfra.sagemakerExecutionRoleArn,
});
neptuneStack.addDependency(smInfra);

// 5b. SageMaker Domain (Studio + JupyterServer notebook)
const domainStack = new SageMakerDomainStack(app, "SageMakerDomainStack", {
  env: env,
  domainName: "fraud-detection-domain",
  executionRoleArn: smInfra.sagemakerExecutionRoleArn,
  notebookRepoUrl: repoUrl,
  notebookRepoBranch: branch,
});
domainStack.addDependency(smInfra);

// 5c. VPC peering: SageMaker Studio ↔ Neptune
const peeringStack = new VpcPeeringStack(app, "VpcPeeringStack", {
  env: env,
  sagemakerVpc: domainStack.vpc,
  neptuneVpc: neptuneStack.vpc,
  neptuneSecurityGroup: neptuneStack.securityGroup,
  neptunePort: neptuneStack.clusterPortNumber,
});
peeringStack.addDependency(domainStack);
peeringStack.addDependency(neptuneStack);

// 6. Triton Endpoint (SageMaker)
// Note: This requires a model.tar.gz to exist in the model bucket at the specified path.
// This stack is typically deployed AFTER the training pipeline has run at least once.
const endpointStack = new SageMakerTritonEndpointStack(
  app,
  "SageMakerTritonEndpointStack",
  {
    env: env,
    tritonImageUri: `${tritonImageRepo.repositoryUri}:latest`,
    modelDataUrl: `s3://${modelBucketName}/model-repository/model.tar.gz`,
    executionRoleArn: smInfra.sagemakerExecutionRoleArn,
  },
);
endpointStack.addDependency(smInfra);
endpointStack.addDependency(tritonImageRepo);
