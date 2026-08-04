import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";

export interface VpcPeeringStackProps extends cdk.StackProps {
  /** The SageMaker Studio VPC (requester side). */
  readonly sagemakerVpc: ec2.Vpc;
  /** The Neptune VPC (accepter side). */
  readonly neptuneVpc: ec2.Vpc;
  /** Neptune security group — an ingress rule will be added for the SM CIDR. */
  readonly neptuneSecurityGroup: ec2.SecurityGroup;
  /** Neptune port number (default 8182). */
  readonly neptunePort?: number;
}

/**
 * Creates a VPC peering connection between the SageMaker Studio VPC and the
 * Neptune VPC, adds the required route-table entries in both directions, and
 * opens the Neptune security group to traffic from the SageMaker CIDR.
 */
export class VpcPeeringStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: VpcPeeringStackProps) {
    super(scope, id, props);

    const neptunePort = props.neptunePort ?? 8182;

    // ─── Peering connection ────────────────────────────────────────────
    const peering = new ec2.CfnVPCPeeringConnection(this, "SmToNeptunePeering", {
      vpcId: props.sagemakerVpc.vpcId,
      peerVpcId: props.neptuneVpc.vpcId,
      tags: [{ key: "Name", value: "sagemaker-to-neptune" }],
    });

    // ─── Routes: SageMaker private subnets → Neptune VPC CIDR ─────────
    props.sagemakerVpc.privateSubnets.forEach((subnet, idx) => {
      new ec2.CfnRoute(this, `SmToNeptuneRoute${idx}`, {
        routeTableId: subnet.routeTable.routeTableId,
        destinationCidrBlock: props.neptuneVpc.vpcCidrBlock,
        vpcPeeringConnectionId: peering.attrId,
      });
    });

    // ─── Routes: Neptune private subnets → SageMaker VPC CIDR ─────────
    props.neptuneVpc.privateSubnets.forEach((subnet, idx) => {
      new ec2.CfnRoute(this, `NeptuneToSmRoute${idx}`, {
        routeTableId: subnet.routeTable.routeTableId,
        destinationCidrBlock: props.sagemakerVpc.vpcCidrBlock,
        vpcPeeringConnectionId: peering.attrId,
      });
    });

    // ─── Security group: allow SageMaker CIDR → Neptune port ──────────
    props.neptuneSecurityGroup.addIngressRule(
      ec2.Peer.ipv4(props.sagemakerVpc.vpcCidrBlock),
      ec2.Port.tcp(neptunePort),
      "Neptune access from SageMaker Studio VPC via peering",
    );

    // ─── Outputs ──────────────────────────────────────────────────────
    new cdk.CfnOutput(this, "PeeringConnectionId", {
      value: peering.attrId,
      description: "VPC Peering Connection ID (SageMaker ↔ Neptune)",
    });
  }
}
