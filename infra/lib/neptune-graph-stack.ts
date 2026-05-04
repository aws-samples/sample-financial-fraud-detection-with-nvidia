import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as neptune from '@aws-cdk/aws-neptune-alpha';
import { Construct } from 'constructs';

export interface NeptuneGraphStackProps extends cdk.StackProps {
  sagemakerExecutionRole: iam.IRole;
}

export class NeptuneGraphStack extends cdk.Stack {
  public readonly clusterEndpoint: string;
  public readonly clusterPort: string;
  public readonly vpc: ec2.IVpc;
  public readonly securityGroup: ec2.ISecurityGroup;

  constructor(scope: Construct, id: string, props: NeptuneGraphStackProps) {
    super(scope, id, props);

    this.vpc = new ec2.Vpc(this, 'NeptuneVpc', {
      maxAzs: 2,
      natGateways: 1,
    });

    const cluster = new neptune.DatabaseCluster(this, 'NeptuneCluster', {
      vpc: this.vpc,
      instanceType: neptune.InstanceType.SERVERLESS,
      iamAuthentication: true,
      serverlessScalingConfiguration: {
        minCapacity: 1,
        maxCapacity: 8,
      },
      storageEncrypted: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    cluster.connections.allowDefaultPortFrom(
      ec2.Peer.ipv4(this.vpc.vpcCidrBlock),
      'Neptune access from VPC',
    );

    cluster.grantConnect(props.sagemakerExecutionRole);

    this.securityGroup = cluster.connections.securityGroups[0];
    this.clusterEndpoint = cluster.clusterEndpoint.hostname;
    this.clusterPort = cluster.clusterEndpoint.port.toString();

    new cdk.CfnOutput(this, 'NeptuneEndpoint', {
      value: cluster.clusterEndpoint.hostname,
      exportName: 'NeptuneClusterEndpoint',
    });
    new cdk.CfnOutput(this, 'NeptunePort', {
      value: cluster.clusterEndpoint.port.toString(),
      exportName: 'NeptuneClusterPort',
    });
    new cdk.CfnOutput(this, 'VpcId', {
      value: this.vpc.vpcId,
      exportName: 'NeptuneVpcId',
    });
  }
}
