import * as cdk from "aws-cdk-lib";
import * as sagemaker from "aws-cdk-lib/aws-sagemaker";
import * as applicationautoscaling from "aws-cdk-lib/aws-applicationautoscaling";
import { Construct } from "constructs";

export interface SageMakerTritonEndpointStackProps extends cdk.StackProps {
    /**
     * The ECR image URI for the custom Triton container
     */
    tritonImageUri: string;

    /**
     * The S3 bucket name where the model artifacts are stored
     * The path is expected to be s3://<bucket>/model.tar.gz
     */
    modelDataUrl: string;

    /**
     * The IAM Role ARN for SageMaker execution
     */
    executionRoleArn: string;

    /**
     * Instance type for the endpoint
     * @default "ml.g4dn.2xlarge"
     */
    instanceType?: string;
}

export class SageMakerTritonEndpointStack extends cdk.Stack {
    public readonly endpointName: string;

    constructor(scope: Construct, id: string, props: SageMakerTritonEndpointStackProps) {
        super(scope, id, props);

        const instanceType = props.instanceType ?? "ml.g4dn.2xlarge";

        // 1. SageMaker Model
        const model = new sagemaker.CfnModel(this, "TritonModel", {
            executionRoleArn: props.executionRoleArn,
            primaryContainer: {
                image: props.tritonImageUri,
                modelDataUrl: props.modelDataUrl,
                environment: {
                    SAGEMAKER_PROGRAM: "serve",
                    SAGEMAKER_SUBMIT_DIRECTORY: "/opt/ml/model/code",
                },
            },
        });

        // 2. Endpoint Configuration
        const endpointConfig = new sagemaker.CfnEndpointConfig(this, "TritonEndpointConfig", {
            productionVariants: [
                {
                    initialInstanceCount: 1,
                    instanceType: instanceType,
                    modelName: model.attrModelName,
                    variantName: "AllTraffic",
                },
            ],
        });

        // 3. Endpoint
        const endpoint = new sagemaker.CfnEndpoint(this, "TritonEndpoint", {
            endpointConfigName: endpointConfig.attrEndpointConfigName,
        });

        this.endpointName = endpoint.attrEndpointName;

        // 4. Auto-scaling
        const resourceId = `endpoint/${endpoint.attrEndpointName}/variant/AllTraffic`;

        const target = new applicationautoscaling.ScalableTarget(this, "TritonScalableTarget", {
            serviceNamespace: applicationautoscaling.ServiceNamespace.SAGEMAKER,
            resourceId: resourceId,
            scalableDimension: "sagemaker:variant:DesiredInstanceCount",
            minCapacity: 1,
            maxCapacity: 3,
        });

        // Scale on InvocationsPerInstance
        target.scaleToTrackMetric("TargetTrackingScaling", {
            targetValue: 100, // Target 100 invocations per minute per instance
            predefinedMetric: applicationautoscaling.PredefinedMetric.SAGEMAKER_VARIANT_INVOCATIONS_PER_INSTANCE,
            scaleInCooldown: cdk.Duration.seconds(300),
            scaleOutCooldown: cdk.Duration.seconds(60),
        });

        // Outputs
        new cdk.CfnOutput(this, "TritonEndpointName", {
            value: this.endpointName,
            exportName: "TritonEndpointName",
        });

        new cdk.CfnOutput(this, "TritonEndpointUrl", {
            value: `https://runtime.sagemaker.${this.region}.amazonaws.com/endpoints/${this.endpointName}/invocations`,
            exportName: "TritonEndpointUrl",
        });
    }
}
