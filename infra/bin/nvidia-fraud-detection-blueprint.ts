#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { AwsSolutionsChecks } from "cdk-nag";
import { NvidiaFraudDetectionBlueprint } from "../lib/nvidia-fraud-detection-blueprint";
import { TarExtractorStack } from "../lib/tar-extractor-stack";
import { SageMakerExecutionRoleStack } from "../lib/sagemaker-training-role";
import { BlueprintECRStack } from "../lib/training-image-repo";

const app = new cdk.App();

// Add CDK Nag checks
cdk.Aspects.of(app).add(new AwsSolutionsChecks({ verbose: true }));

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION,
};

const modelBucketName = "ml-on-containers-" + process.env.CDK_DEFAULT_ACCOUNT;

const tarExtractorStack = new TarExtractorStack(
  app,
  "NvidiaFraudDetectionBlueprintModelExtractor",
  {
    env: env,
    modelBucketName: modelBucketName,
    synthesizer: new cdk.DefaultStackSynthesizer({
      qualifier: "nvidia", // Must match the qualifier used in bootstrap
    }),
  },
);

const sagemakerExecutionRole = new SageMakerExecutionRoleStack(
  app,
  "NvidiaFraudDetectionTrainingRole",
  {
    env: env,
    modelBucketArn: "arn:aws:s3:::" + modelBucketName,
    synthesizer: new cdk.DefaultStackSynthesizer({
      qualifier: "nvidia", // Must match the qualifier used in bootstrap
    }),
  },
);

const trainingImageRepo = new BlueprintECRStack(
  app,
  "NvidiaFraudDetectionTrainingImageRepo",
  {
    env: env,
    synthesizer: new cdk.DefaultStackSynthesizer({
      qualifier: "nvidia", // Must match the qualifier used in bootstrap
    }),
  },
);

const mainStack = new NvidiaFraudDetectionBlueprint(
  app,
  "NvidiaFraudDetectionBlueprint",
  {
    env: env,
    modelBucketName: modelBucketName + "-model-registry",
    synthesizer: new cdk.DefaultStackSynthesizer({
      qualifier: "nvidia", // Must match the qualifier used in bootstrap
    }),
  },
);

mainStack.addDependency(trainingImageRepo);
mainStack.addDependency(sagemakerExecutionRole);
mainStack.addDependency(tarExtractorStack);

app.synth();
