import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as eks from 'aws-cdk-lib/aws-eks';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as blueprints from '@aws-quickstart/eks-blueprints'
import { Construct } from 'constructs';
import { ALBDefaultIngressClassAddOn } from './alb-default-ingress-class-addon';

export interface NvidiaFraudDetectionBlueprintProps extends cdk.StackProps {
  /**
   * The S3 bucket name where models are stored
   */
  modelBucketName: string;

}

export class NvidiaFraudDetectionBlueprint extends cdk.Stack {
  constructor(scope: Construct, id: string, props: NvidiaFraudDetectionBlueprintProps) {
    super(scope, id, props);

    // Use existing VPC or create a new one
    const vpc = new ec2.Vpc(this, 'TritonVpc', {
      maxAzs: 3,
      natGateways: 1,
      enableDnsHostnames: true,
      enableDnsSupport: true,
    });

    const g4dnNodePoolSpec: blueprints.NodePoolV1Spec = {
      labels: {
        "node-type": "gpu",
        "instance-family": "g4dn",
        "nvidia.com/gpu": "true",
        "workload": "ml-inference"
      },

      // Taints to ensure only GPU workloads schedule on these nodes
      taints: [
        {
          key: "nvidia.com/gpu",
          value: "Exists",
          effect: "NoSchedule"
        },
      ],

      // Startup taints during node initialization
      startupTaints: [
        {
          key: "node.kubernetes.io/not-ready",
          effect: "NoSchedule"
        }
      ],

      // Requirements for g4dn instance selection
      requirements: [
        { key: "karpenter.sh/capacity-type", operator: "In", values: ["on-demand"] },
        {
          key: "node.kubernetes.io/instance-type", operator: "In", values: [
            "g4dn.xlarge",    // 1 GPU, 4 vCPUs, 16 GB
            "g4dn.2xlarge",   // 1 GPU, 8 vCPUs, 32 GB
          ],
        },
        { key: "kubernetes.io/arch", operator: "In", values: ["amd64"] },
        { key: "topology.kubernetes.io/zone", operator: "In", values: [`${props.env?.region}a`, `${props.env?.region}b`, `${props.env?.region}c`] }
      ],

      // Node lifecycle - expire after 24h for cost optimization
      expireAfter: "24h",

      // Disruption settings for GPU workloads
      disruption: {
        consolidationPolicy: "WhenEmpty", // Conservative for GPU workloads
        consolidateAfter: "30s"
      },

      // Resource limits for the pool
      limits: {
        cpu: 320,           // Max ~5 g4dn.16xlarge instances
        memory: "1280Gi",   // Max memory across instances
        "nvidia.com/gpu": 8 // Max 8 GPUs total
      },

      // Higher priority for GPU nodes
      weight: 100
    };

    const triton = new blueprints.teams.ApplicationTeam({
      name: "triton",
      namespace: "triton",
      serviceAccountName: "triton-sa",
      serviceAccountPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3ReadOnlyAccess')]
    })


    const repoUrl = "git@ssh.gitlab.aws.dev:ahsanrm/nvidia-fraud-detection-blueprint-on-aws.git";


    const addons = [
      new ALBDefaultIngressClassAddOn(),
      new blueprints.GpuOperatorAddon({
        version: "v25.3.2"
      }),
      new blueprints.SecretsStoreAddOn(),
      new blueprints.ArgoCDAddOn({
        bootstrapRepo: {
          repoUrl: repoUrl,
          targetRevision: 'main',
          path: 'infra/manifests/argocd',
          credentialsSecretName: "nvidia-gitlab-token", // Doesnt work with gitlab - needs to be resolved
          credentialsType: 'TOKEN'
        },
        bootstrapValues: {
          repoUrl: repoUrl,
          namespace: "triton",
          serviceAccount: {
            name: "triton-sa"
          },
          targetRevision: "main",
          account: this.account,
          region: this.region,
          bucketName: props.modelBucketName
        }
      }),
    ];

    blueprints.AutomodeBuilder.builder({
      extraNodePools: {
        'nvidia': g4dnNodePoolSpec
      }
    }).account(this.account)
      .region(this.region)
      .version(eks.KubernetesVersion.V1_32)
      .addOns(...addons)
      .teams(triton)
      .resourceProvider(blueprints.GlobalResources.Vpc, new blueprints.DirectVpcProvider(vpc))
      .build(this, "ClusterBlueprint", {
        synthesizer: new cdk.DefaultStackSynthesizer({
          qualifier: 'nvidia', // Must match the qualifier used in bootstrap
        }),
      });

  }
}
