import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as eks from "aws-cdk-lib/aws-eks";
import * as iam from "aws-cdk-lib/aws-iam";
import * as blueprints from "@aws-quickstart/eks-blueprints";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { KfAddon } from "./kf-addon";
import { argoCdValues } from "./argocd-values";
import { NodePoolAddon } from "./nodepool-addon";

export interface NvidiaFraudDetectionBlueprintProps extends cdk.StackProps {
  /**
   * The S3 bucket name where models are stored
   */
  modelBucketName: string;

  kubeflowBucketName: string;

  /**
   * The S3 bucket name for raw data (e.g., TabFormer dataset)
   */
  dataBucketName: string;

  /**
   * The S3 bucket name for model registry (Triton model repository)
   */
  modelRegistryBucketName: string;

  /**
   * The Triton inference server image URI (from ECR)
   */
  tritonImageUri: string;

  /**
   * The name of the NGC API Key Secret in SecretsManager
   */
  ngcSecretName: string;

  /**
   * The hostname in route53
   */
  hostname: string;

}

export class NvidiaFraudDetectionBlueprint extends cdk.Stack {
  constructor(
    scope: Construct,
    id: string,
    props: NvidiaFraudDetectionBlueprintProps,
  ) {
    super(scope, id, props);

    // Use existing VPC or create a new one
    const vpc = new ec2.Vpc(this, "TritonVpc", {
      maxAzs: 3,
      natGateways: 1,
      enableDnsHostnames: true,
      enableDnsSupport: true,
      flowLogs: {
        VpcFlowLog: {
          destination: ec2.FlowLogDestination.toCloudWatchLogs(),
        },
      },
    });

    const kubeflowBucket = new s3.Bucket(this, "KubeflowPipelinesBucket", {
      bucketName: props.kubeflowBucketName,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      serverAccessLogsPrefix: "access-logs/",
      enforceSSL: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
    });

    const hostedZoneProvider = new blueprints.LookupHostedZoneProvider(
      props.hostname,
    );

    const g4dnNodePoolSpec: blueprints.NodePoolV1Spec = {
      labels: {
        "node-type": "gpu",
        "instance-family": "g4dn",
        "nvidia.com/gpu": "true",
        workload: "ml-inference",
      },

      // Taints to ensure only GPU workloads schedule on these nodes
      taints: [
        {
          key: "nvidia.com/gpu",
          value: "true",
          effect: "NoSchedule",
        },
      ],

      // Startup taints during node initialization
      startupTaints: [
        {
          key: "node.kubernetes.io/not-ready",
          effect: "NoSchedule",
        },
      ],

      // Requirements for g4dn instance selection
      requirements: [
        {
          key: "karpenter.sh/capacity-type",
          operator: "In",
          values: ["on-demand"],
        },
        {
          key: "node.kubernetes.io/instance-type",
          operator: "In",
          values: [
            "g4dn.2xlarge", // 1 GPU, 8 vCPUs, 32 GB - minimum for 24M row cuDF
          ],
        },
        { key: "kubernetes.io/arch", operator: "In", values: ["amd64"] },
        {
          key: "topology.kubernetes.io/zone",
          operator: "In",
          values: [
            `${props.env?.region}a`,
            `${props.env?.region}b`,
            `${props.env?.region}c`,
          ],
        },
        // Ensure GPU-accelerated AMI family is used
        {
          key: "karpenter.k8s.aws/instance-gpu-count",
          operator: "Gt",
          values: ["0"],
        },
      ],

      // Node lifecycle - expire after 24h for cost optimization
      expireAfter: "24h",

      // Disruption settings for GPU workloads
      disruption: {
        consolidationPolicy: "WhenEmpty", // Conservative for GPU workloads
        consolidateAfter: "30s",
      },

      // Resource limits for the pool
      limits: {
        cpu: 320, // Max ~5 g4dn.16xlarge instances
        memory: "1280Gi", // Max memory across instances
        "nvidia.com/gpu": 8, // Max 8 GPUs total
      },

      // Higher priority for GPU nodes
      weight: 100,
    };

    const triton = new blueprints.teams.ApplicationTeam({
      name: "triton",
      namespace: "triton",
      serviceAccountName: "triton-sa",
      serviceAccountPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonS3ReadOnlyAccess"),
      ],
    });

    const repoUrl =
      "https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia";

    Reflect.defineMetadata("ordered", true, blueprints.addons.ArgoCDAddOn);
    Reflect.defineMetadata("ordered", true, blueprints.addons.KarpenterV1AddOn);

    const addons = [
      new blueprints.addons.KarpenterV1AddOn({
        nodePoolSpec: g4dnNodePoolSpec,
        ec2NodeClassSpec: {
          amiFamily: "AL2023",
          amiSelectorTerms: [{ alias: "al2023@latest" }],
          subnetSelectorTerms: [
            {
              tags: {
                Name: "*Private*",
              },
            },
          ],
          securityGroupSelectorTerms: [
            {
              tags: {
                "aws:eks:cluster-name":
                  "nvidia-fraud-detection-cluster-blueprint",
              },
            },
          ],
          blockDeviceMappings: [
            {
              deviceName: "/dev/xvda",
              ebs: {
                volumeSize: "200Gi",
                deleteOnTermination: true,
              },
            },
          ],
        },
      }),
      new NodePoolAddon({
        nodePoolSpec: {
          labels: {
            "node-type": "general-purpose",
            "instance-family": "m5",
          },
          requirements: [
            {
              key: "karpenter.sh/capacity-type",
              operator: "In",
              values: ["on-demand"],
            },
            {
              key: "node.kubernetes.io/instance-type",
              operator: "In",
              values: ["m5.2xlarge"],
            },
            { key: "kubernetes.io/arch", operator: "In", values: ["amd64"] },
          ],
          expireAfter: "24h",
          disruption: {
            consolidationPolicy: "WhenEmpty",
            consolidateAfter: "30s",
          },
          limits: {
            cpu: 64,
            memory: "256Gi",
          },
          weight: 10,
        },
      }),
      new blueprints.addons.AwsLoadBalancerControllerAddOn(),
      new blueprints.addons.GpuOperatorAddon({
        version: "v25.3.2",
      }),
      new blueprints.addons.SecretsStoreAddOn(),
      new blueprints.addons.ExternalsSecretsAddOn(),
      new blueprints.addons.ExternalDnsAddOn({
        hostedZoneResources: [blueprints.GlobalResources.HostedZone],
        sources: ["istio-gateway"],
      }),
      new blueprints.addons.EbsCsiDriverAddOn({ storageClass: "gp3" }),
      new blueprints.addons.ArgoCDAddOn({
        bootstrapRepo: {
          repoUrl: repoUrl,
          targetRevision: "v2",
          path: "infra/manifests/argocd",
        },
        bootstrapValues: {
          repoUrl: repoUrl,
          namespace: "triton",
          serviceAccount: {
            name: "triton-sa",
          },
          targetRevision: "v2",
          account: this.account,
          region: this.region,
          bucketName: props.modelBucketName,
          image: {
            imageName: props.tritonImageUri,
          },
          ngcSecretName: props.ngcSecretName,
        },
        values: argoCdValues,
      }),
      new KfAddon({
        bucketName: props.kubeflowBucketName,
        dataBucketName: props.dataBucketName,
        modelBucketName: props.modelRegistryBucketName,
        hostname: props.hostname,
        email: "zjacobso@amazon.com"
      }),
    ];

    // Use EKS with Karpenter instead of Automode for GPU driver flexibility
    const cluster = blueprints.EksBlueprint.builder()
      .account(this.account)
      .region(this.region)
      .version(eks.KubernetesVersion.V1_33)
      .addOns(...addons)
      .teams(triton)
      .resourceProvider(
        blueprints.GlobalResources.Vpc,
        new blueprints.DirectVpcProvider(vpc),
      )
      .resourceProvider(
        blueprints.GlobalResources.HostedZone,
        hostedZoneProvider,
      )
      .clusterProvider(
        new blueprints.MngClusterProvider({
          amiType: eks.NodegroupAmiType.AL2023_X86_64_STANDARD,
          minSize: 2,
          desiredSize: 3,
          maxSize: 5,
          instanceTypes: [
            ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.XLARGE),
          ],
        }),
      )
      .build(this, "nvidia-fraud-detection-cluster-blueprint");
  }
}
