import * as cdk from "aws-cdk-lib";
import { NvidiaFraudDetectionBlueprint } from "../lib/nvidia-fraud-detection-blueprint";
import { SageMakerTrainingImageRepoStack } from "../lib/sagemaker-training-image-repo";
import { SageMakerPreprocessingImageRepoStack } from "../lib/sagemaker-preprocessing-image-repo";
import { TritonImageRepoStack } from "../lib/triton-image-repo";
import { SageMakerInfraStack } from "../lib/sagemaker-infrastructure-stack";
import { SageMakerTritonEndpointStack } from "../lib/sagemaker-triton-endpoint-stack";
import { SageMakerDomainStack } from "../lib/sagemaker-domain-stack";

const app = new cdk.App();

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION,
};

// Config
const ngcSecretName = app.node.tryGetContext("ngcSecretName") || "ngc-api-key";
// Use sm suffix (sm = sagemaker) to distinguish from old kubeflow buckets
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
    repoUrl:
      "https://github.com/atroyanovsky/TW-sample-financial-fraud-detection-with-nvidia.git",
    branch: "v2_sagemaker",
  },
);

const preprocessingImageRepo = new SageMakerPreprocessingImageRepoStack(
  app,
  "SageMakerPreprocessingImageRepoStack",
  {
    env: env,
    repoUrl:
      "https://github.com/atroyanovsky/TW-sample-financial-fraud-detection-with-nvidia.git",
    branch: "v2_sagemaker",
  },
);

// 2. Inference Image Repo (Triton)
const tritonImageRepo = new TritonImageRepoStack(app, "TritonImageRepoStack", {
  env: env,
  repoUrl:
    "https://github.com/atroyanovsky/TW-sample-financial-fraud-detection-with-nvidia.git",
  branch: "v2_sagemaker",
});

// 3. Base Infrastructure (VPC, S3)
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

// 5. SageMaker Domain (for Studio access to Pipelines)
const domainStack = new SageMakerDomainStack(app, "SageMakerDomainStack", {
  env: env,
  domainName: "fraud-detection-domain",
  executionRoleArn: smInfra.sagemakerExecutionRoleArn,
});
domainStack.addDependency(smInfra);

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
