import { ArgoCDAddOn, ClusterAddOn, ClusterInfo } from "@aws-quickstart/eks-blueprints";
import { createServiceAccount, dependable } from "@aws-quickstart/eks-blueprints/dist/utils";
import { Construct } from "constructs";
import * as iam from "aws-cdk-lib/aws-iam";
import { CfnJson } from "aws-cdk-lib";
import { deployKfApp } from "./deploykf-app";
import { KfIAMPolicy } from "./kf-policy";
import { KubernetesObjectValue } from "aws-cdk-lib/aws-eks";

export interface KfAddonProps {
  bucketName: string;
  dataBucketName: string;
  modelBucketName: string;
  hostname: string;
  email: string;
}

export class KfAddon implements ClusterAddOn {
  constructor(readonly props: KfAddonProps) { }

  @dependable("ArgoCDAddOn")
  deploy(clusterInfo: ClusterInfo): Promise<Construct> | void {
    const kfPolicyDoc = iam.PolicyDocument.fromJson(
      KfIAMPolicy(this.props.bucketName, this.props.dataBucketName, this.props.modelBucketName)
    );

    const kfPolicy = new iam.ManagedPolicy(clusterInfo.cluster, "KubeflowPolicy", { document: kfPolicyDoc });

    const conditions = new CfnJson(clusterInfo.cluster, 'ConditionJson', {
      value: {
        [`${clusterInfo.cluster.openIdConnectProvider.openIdConnectProviderIssuer}:aud`]: 'sts.amazonaws.com',
        [`${clusterInfo.cluster.openIdConnectProvider.openIdConnectProviderIssuer}:sub`]: `system:serviceaccount:*`,
      },
    });
    const principal = new iam.OpenIdConnectPrincipal(clusterInfo.cluster.openIdConnectProvider).withConditions({
      StringLike: conditions,
    });
    const kfRole = new iam.Role(clusterInfo.cluster, 'kf-role', { assumedBy: principal });
    kfRole.addManagedPolicy(kfPolicy);

    const route53PolicyDoc = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ['route53:GetChange'],
          resources: ['arn:aws:route53:::change/*']
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ['route53:ChangeResourceRecordSets', 'route53:ListResourceRecordSets'],
          resources: ['arn:aws:route53:::hostedzone/*'],
          conditions: {
            'ForAllValues:StringEquals': {
              'route53:ChangeResourceRecordSetsRecordTypes': ['TXT']
            }
          }
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ['route53:ListHostedZonesByName'],
          resources: ['*']
        })
      ]
    });


    const certManagerSA = createServiceAccount(clusterInfo.cluster, "cert-manager-acme-sa", "cert-manager", route53PolicyDoc)

    const argoCrdCheck = new KubernetesObjectValue(clusterInfo.cluster, 'ArgoCRDCheck', {
      cluster: clusterInfo.cluster,
      objectType: 'crd',
      objectName: 'applications.argoproj.io',
      jsonPath: '.status.conditions[?(@.type=="Established")].status'
    });

    const deployKfManifest = clusterInfo.cluster.addManifest("deployKF-argo-app", deployKfApp(this.props.bucketName, clusterInfo.cluster.stack.region, kfRole.roleArn, certManagerSA.role.roleArn, this.props.hostname, this.props.email, certManagerSA.serviceAccountName))
    deployKfManifest.node.addDependency(argoCrdCheck)
    return Promise.resolve(deployKfManifest)
  }

}
