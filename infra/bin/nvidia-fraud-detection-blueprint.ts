import * as cdk from "aws-cdk-lib";
import { NvidiaFraudDetectionBlueprint } from "../lib/nvidia-fraud-detection-blueprint";
import { BlueprintECRStack } from "../lib/training-image-repo";
import { TritonImageRepoStack } from "../lib/triton-image-repo";

const app = new cdk.App();

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION,
};

const ngcSecretName = app.node.tryGetContext("ngcSecretName") || "ngc-api-key";
const hostname = app.node.tryGetContext("hostname");

const modelBucketName = "ml-on-containers-" + process.env.CDK_DEFAULT_ACCOUNT;
const kfBucketName = "kubeflow-pipelines-" + process.env.CDK_DEFAULT_ACCOUNT;
const dataBucketName = modelBucketName;
const modelRegistryBucketName = modelBucketName + "-model-registry";

const trainingImageRepo = new BlueprintECRStack(
  app,
  "NvidiaFraudDetectionTrainingImageRepo",
  {
    env: env,
  },
);

const tritonImageRepo = new TritonImageRepoStack(
  app,
  "NvidiaFraudDetectionTritonImageRepo",
  { env: env },
);

const mainStack = new NvidiaFraudDetectionBlueprint(
  app,
  "NvidiaFraudDetectionBlueprint",
  {
    env: env,
    modelBucketName: modelRegistryBucketName,
    kubeflowBucketName: kfBucketName,
    dataBucketName: dataBucketName,
    modelRegistryBucketName: modelRegistryBucketName,
    tritonImageUri: `${tritonImageRepo.repositoryUri}:latest`,
    hostname: hostname,
    ngcSecretName: ngcSecretName,
  },
);

mainStack.addDependency(trainingImageRepo);
mainStack.addDependency(tritonImageRepo);
