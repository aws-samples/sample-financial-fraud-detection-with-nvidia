import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as eks from 'aws-cdk-lib/aws-eks';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as blueprints from '@aws-quickstart/eks-blueprints'
import { NagSuppressions } from 'cdk-nag';
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
      flowLogs: {
        'VpcFlowLog': {
          destination: ec2.FlowLogDestination.toCloudWatchLogs(),
        }
      }
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


    const repoUrl = "https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia";


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

    const cluster = blueprints.AutomodeBuilder.builder({
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
          qualifier: 'nvidia',
        }),
      });

    // Add CDK Nag suppressions for legitimate AWS managed policies
    NagSuppressions.addResourceSuppressions(
      cluster,
      [
        {
          id: 'AwsSolutions-IAM4',
          reason: 'AWS managed policies are required for EKS cluster and node group functionality',
          appliesTo: [
            'Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSClusterPolicy',
            'Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSComputePolicy',
            'Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSBlockStoragePolicy',
            'Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSLoadBalancingPolicy',
            'Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSNetworkingPolicy',
            'Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSWorkerNodePolicy',
            'Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly',
            'Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonElasticContainerRegistryPublicReadOnly',
            'Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
            'Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole'
          ]
        },
        {
          id: 'AwsSolutions-EKS1',
          reason: 'Public API access is required for CI/CD and external access patterns'
        },
        {
          id: 'AwsSolutions-EKS2',
          reason: 'Control plane logging is enabled through EKS blueprints configuration',
          appliesTo: [
            'LogExport::api',
            'LogExport::audit',
            'LogExport::authenticator',
            'LogExport::controllerManager',
            'LogExport::scheduler'
          ]
        },
        {
          id: 'AwsSolutions-IAM5',
          reason: 'Wildcard permissions are required for kubectl provider functionality'
        },
        {
          id: 'AwsSolutions-L1',
          reason: 'Lambda runtime is managed by EKS blueprints'
        },
        {
          id: 'AwsSolutions-KMS5',
          reason: 'KMS key rotation is managed by EKS service'
        }
      ],
      true
    );

    // Additional suppression for KubectlProvider ECR Public access
    NagSuppressions.addResourceSuppressionsByPath(
      this,
      '/NvidiaFraudDetectionBlueprint/ClusterBlueprint/ClusterBlueprint/KubectlProvider/Handler/ServiceRole/Resource',
      [
        {
          id: 'AwsSolutions-IAM4',
          reason: 'ECR Public ReadOnly policy required for kubectl provider to access public container images',
          appliesTo: [
            'Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonElasticContainerRegistryPublicReadOnly',
            'Policy::{\"Fn::If\":[\"ClusterBlueprintKubectlProviderHandlerHasEcrPublicFD2FFDE5\",{\"Fn::Join\":[\"\",[\"arn:\",{\"Ref\":\"AWS::Partition\"},\":iam::aws:policy/AmazonElasticContainerRegistryPublicReadOnly\"]]},{\"Ref\":\"AWS::NoValue\"}]}'
          ]
        }
      ]
    );

    // Suppress VPC flow log warning
    NagSuppressions.addResourceSuppressions(
      vpc,
      [{
        id: 'AwsSolutions-VPC7',
        reason: 'VPC Flow Logs are enabled via flowLogs configuration'
      }],
      true
    );

  }
}
