import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as neptune from 'aws-cdk-lib/aws-neptune';
import * as iam from 'aws-cdk-lib/aws-iam';
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

    this.securityGroup = new ec2.SecurityGroup(this, 'NeptuneSG', {
      vpc: this.vpc,
      description: 'Security group for Neptune cluster',
      allowAllOutbound: true,
    });
    this.securityGroup.addIngressRule(
      ec2.Peer.ipv4(this.vpc.vpcCidrBlock),
      ec2.Port.tcp(8182),
      'Neptune access from VPC',
    );

    const subnetGroup = new neptune.CfnDBSubnetGroup(this, 'NeptuneSubnetGroup', {
      dbSubnetGroupDescription: 'Subnet group for Neptune fraud detection cluster',
      subnetIds: this.vpc.privateSubnets.map(s => s.subnetId),
    });

    const cluster = new neptune.CfnDBCluster(this, 'NeptuneCluster', {
      dbClusterIdentifier: 'fraud-detection-neptune',
      engineVersion: '1.3.2.1',
      dbSubnetGroupName: subnetGroup.ref,
      vpcSecurityGroupIds: [this.securityGroup.securityGroupId],
      iamAuthEnabled: true,
      storageEncrypted: true,
      serverlessScalingConfiguration: {
        minCapacity: 1,
        maxCapacity: 8,
      },
    });
    cluster.addDependency(subnetGroup);

    const instance = new neptune.CfnDBInstance(this, 'NeptuneInstance', {
      dbInstanceClass: 'db.serverless',
      dbClusterIdentifier: cluster.ref,
      dbInstanceIdentifier: 'fraud-detection-neptune-instance',
    });
    instance.addDependency(cluster);

    // Grant SageMaker role access to Neptune
    props.sagemakerExecutionRole.addToPolicy(new iam.PolicyStatement({
      actions: ['neptune-db:*'],
      resources: [`arn:aws:neptune-db:${this.region}:${this.account}:${cluster.attrClusterResourceId}/*`],
    }));

    this.clusterEndpoint = cluster.attrEndpoint;
    this.clusterPort = cluster.attrPort;

    new cdk.CfnOutput(this, 'NeptuneEndpoint', {
      value: cluster.attrEndpoint,
      exportName: 'NeptuneClusterEndpoint',
    });
    new cdk.CfnOutput(this, 'NeptunePort', {
      value: cluster.attrPort,
      exportName: 'NeptuneClusterPort',
    });
    new cdk.CfnOutput(this, 'VpcId', {
      value: this.vpc.vpcId,
      exportName: 'NeptuneVpcId',
    });
  }
}
