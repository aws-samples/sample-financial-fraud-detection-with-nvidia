import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
imoprt * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

export interface BlueprintECRStackProps extends cdk.StackProps {

}

export class BlueprintECRStack extends cdk.Stack {

  constructor(
    scope: Construct,
    id: string,
    props?: BlueprintECRStackProps,
  ) {
    super(scope, id, props);
  }
}
