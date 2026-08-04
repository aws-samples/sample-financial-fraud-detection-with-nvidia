import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as neptune from "@aws-cdk/aws-neptune-alpha";
import { Construct } from "constructs";

export interface NeptuneGraphStackProps extends cdk.StackProps {
  sagemakerExecutionRoleArn: string;
}

export class NeptuneGraphStack extends cdk.Stack {
  public readonly clusterEndpoint: string;
  public readonly clusterPort: string;
  public readonly clusterPortNumber: number;
  public readonly vpc: ec2.Vpc;
  public readonly securityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: NeptuneGraphStackProps) {
    super(scope, id, props);

    this.vpc = new ec2.Vpc(this, "NeptuneVpc", {
      ipAddresses: ec2.IpAddresses.cidr("10.1.0.0/16"),
      maxAzs: 2,
      natGateways: 1,
      gatewayEndpoints: {
        S3: { service: ec2.GatewayVpcEndpointAwsService.S3 },
      },
    });

    // Interface endpoint for CloudWatch Logs (SageMaker job logging)
    this.vpc.addInterfaceEndpoint("CloudWatchLogsEndpoint", {
      service: ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
    });

    // Explicit security group so we can expose the concrete L2 type for
    // cross-stack ingress rules (e.g. VPC peering).
    this.securityGroup = new ec2.SecurityGroup(this, "NeptuneSecurityGroup", {
      vpc: this.vpc,
      description: "Security group for Neptune cluster",
      allowAllOutbound: true,
    });

    this.securityGroup.addIngressRule(
      ec2.Peer.ipv4(this.vpc.vpcCidrBlock),
      ec2.Port.tcp(8182),
      "Neptune access from VPC",
    );

    const cluster = new neptune.DatabaseCluster(this, "NeptuneCluster", {
      vpc: this.vpc,
      instanceType: neptune.InstanceType.SERVERLESS,
      iamAuthentication: true,
      securityGroups: [this.securityGroup],
      serverlessScalingConfiguration: {
        minCapacity: 1,
        maxCapacity: 8,
      },
      storageEncrypted: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Create the Neptune access policy locally in this stack to avoid a cyclic
    // cross-stack reference (cluster resource ID must stay in this template).
    const sagemakerRole = iam.Role.fromRoleArn(
      this,
      "ImportedSageMakerRole",
      props.sagemakerExecutionRoleArn,
      {
        mutable: false,
      },
    );

    new iam.Policy(this, "NeptuneConnectPolicy", {
      roles: [sagemakerRole],
      statements: [
        new iam.PolicyStatement({
          actions: ["neptune-db:*"],
          resources: [
            `arn:aws:neptune-db:${this.region}:${this.account}:${cluster.clusterResourceIdentifier}/*`,
          ],
        }),
      ],
    });

    this.clusterEndpoint = cluster.clusterEndpoint.hostname;
    this.clusterPort = cluster.clusterEndpoint.port.toString();
    this.clusterPortNumber = cluster.clusterEndpoint.port;

    new cdk.CfnOutput(this, "NeptuneEndpoint", {
      value: cluster.clusterEndpoint.hostname,
      exportName: "NeptuneClusterEndpoint",
    });
    new cdk.CfnOutput(this, "NeptunePort", {
      value: cluster.clusterEndpoint.port.toString(),
      exportName: "NeptuneClusterPort",
    });
    new cdk.CfnOutput(this, "VpcId", {
      value: this.vpc.vpcId,
      exportName: "NeptuneVpcId",
    });
    new cdk.CfnOutput(this, "NeptuneSubnetIds", {
      value: this.vpc.privateSubnets.map((s) => s.subnetId).join(","),
      exportName: "NeptunePrivateSubnetIds",
    });
    new cdk.CfnOutput(this, "NeptuneSecurityGroupId", {
      value: this.securityGroup.securityGroupId,
      exportName: "NeptuneSecurityGroupId",
    });
  }
}
