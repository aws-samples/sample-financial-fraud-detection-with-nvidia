#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { AwsSolutionsChecks } from 'cdk-nag';
import { NvidiaFraudDetectionBlueprint } from '../lib/nvidia-fraud-detection-blueprint';
import { TarExtractorStack } from '../lib/tar-extractor-stack';
import { SageMakerExecutionRoleStack } from '../lib/sagemaker-training-role';
import { BlueprintECRStack } from '../lib/training-image-repo';
import { S3BucketStack } from '../lib/model-bucket';

const app = new cdk.App();

// Add CDK Nag checks
cdk.Aspects.of(app).add(new AwsSolutionsChecks({ verbose: true }));

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION
}

const modelBucketName = "ml-on-containers-" + process.env.CDK_DEFAULT_ACCOUNT;

const modelBucket = new S3BucketStack(app, 'NvidiaFraudDetectionBlueprintBucket', {
  env: env,
  bucketName: modelBucketName
});

const tarExtractorStack = new TarExtractorStack(app, 'NvidiaFraudDetectionBlueprintModelExtractor', {
  env: env,
  modelBucketName: modelBucketName,
  synthesizer: new cdk.DefaultStackSynthesizer({
    qualifier: 'nvidia', // Must match the qualifier used in bootstrap
  }),
});

const sagemakerExecutionRole = new SageMakerExecutionRoleStack(app, 'NvidiaFraudDetectionTrainingRole', {
  env: env,
  modelBucketArn: "arn:aws:s3:::" + modelBucketName
});

const trainingImageRepo = new BlueprintECRStack(app, 'NvidiaFraudDetectionTrainingImageRepo', {
  env: env
})

const mainStack = new NvidiaFraudDetectionBlueprint(app, 'NvidiaFraudDetectionBlueprint', {
  env: env,
  modelBucketName: modelBucketName + "-model-registry",
  synthesizer: new cdk.DefaultStackSynthesizer({
    qualifier: 'nvidia', // Must match the qualifier used in bootstrap
  }),
});

mainStack.addDependency(tarExtractorStack);

app.synth();
