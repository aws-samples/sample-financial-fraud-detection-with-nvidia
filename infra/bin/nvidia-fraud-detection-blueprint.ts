#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { AwsSolutionsChecks } from 'cdk-nag';
import { NvidiaFraudDetectionBlueprint } from '../lib/nvidia-fraud-detection-blueprint';
import { TarExtractorStack } from '../lib/tar-extractor-stack';

const app = new cdk.App();

// Add CDK Nag checks
cdk.Aspects.of(app).add(new AwsSolutionsChecks({ verbose: true }));

const modelBucketName = "ml-on-containers-gfs"
new NvidiaFraudDetectionBlueprint(app, 'NvidiaFraudDetectionBlueprint', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: 'us-east-1' },
  modelBucketName: modelBucketName + "-model-registry",
  synthesizer: new cdk.DefaultStackSynthesizer({
    qualifier: 'nvidia', // Must match the qualifier used in bootstrap
  }),
});
new TarExtractorStack(app, 'NvidiaFraudDetectionBlueprintModelExtractor', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: 'us-east-1' },
  modelBucketName: modelBucketName,
  synthesizer: new cdk.DefaultStackSynthesizer({
    qualifier: 'nvidia', // Must match the qualifier used in bootstrap
  }),
});

app.synth();