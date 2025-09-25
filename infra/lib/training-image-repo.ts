import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

export interface BlueprintECRStackProps extends cdk.StackProps {}

export class BlueprintECRStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: BlueprintECRStackProps) {
    super(scope, id, props);

    const repo = new ecr.Repository(
      this,
      "NvidiaFraudDetectionBlueprintTrainingRepo",
      {
        repositoryName: "nvidia-training-repo",
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      },
    );

    new cdk.CfnOutput(this, "TrainingImageRepo", {
      value: repo.repositoryName,
    });

    new cdk.CfnOutput(this, "TrainingImageRepoUri", {
      value: repo.repositoryUri
    });
  }
}
