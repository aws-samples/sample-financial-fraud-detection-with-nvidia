import * as cdk from "aws-cdk-lib"
import { HostedZone } from "aws-cdk-lib/aws-route53";
import { Construct } from "constructs";

export interface BlueprintDomainsStackProps extends cdk.StackProps {
  readonly domain: string
}

export class BlueprintDomainStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: BlueprintDomainsStackProps) {
    super(scope, id, props);
    new HostedZone(scope, "HostedZone", { zoneName: props.domain })
  }
}
