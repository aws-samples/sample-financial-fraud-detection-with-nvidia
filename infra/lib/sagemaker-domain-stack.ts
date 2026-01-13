import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

export interface SageMakerDomainStackProps extends cdk.StackProps {
    readonly domainName?: string;
    readonly executionRoleArn: string;
}

export class SageMakerDomainStack extends cdk.Stack {
    public readonly domain: sagemaker.CfnDomain;
    public readonly domainId: string;

    constructor(scope: Construct, id: string, props: SageMakerDomainStackProps) {
        super(scope, id, props);

        const domainName = props.domainName || 'fraud-detection-domain';

        // Create a VPC for SageMaker Studio (required)
        const vpc = new ec2.Vpc(this, 'SageMakerVPC', {
            maxAzs: 2,
            natGateways: 1,
            subnetConfiguration: [
                {
                    cidrMask: 24,
                    name: 'Public',
                    subnetType: ec2.SubnetType.PUBLIC,
                },
                {
                    cidrMask: 24,
                    name: 'Private',
                    subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
                },
            ],
        });

        // Get private subnet IDs
        const subnetIds = vpc.privateSubnets.map(subnet => subnet.subnetId);

        // Create SageMaker Domain
        this.domain = new sagemaker.CfnDomain(this, 'SageMakerDomain', {
            authMode: 'IAM',
            defaultUserSettings: {
                executionRole: props.executionRoleArn,
                securityGroups: [vpc.vpcDefaultSecurityGroup],
            },
            domainName: domainName,
            subnetIds: subnetIds,
            vpcId: vpc.vpcId,
            appNetworkAccessType: 'PublicInternetOnly',
        });

        this.domainId = this.domain.attrDomainId;

        // Create a default user profile
        const userProfile = new sagemaker.CfnUserProfile(this, 'DefaultUserProfile', {
            domainId: this.domainId,
            userProfileName: 'default-user',
            userSettings: {
                executionRole: props.executionRoleArn,
            },
        });

        userProfile.addDependency(this.domain);

        // Outputs
        new cdk.CfnOutput(this, 'DomainId', {
            value: this.domainId,
            description: 'SageMaker Domain ID',
            exportName: `${this.stackName}-DomainId`,
        });

        new cdk.CfnOutput(this, 'DomainArn', {
            value: this.domain.attrDomainArn,
            description: 'SageMaker Domain ARN',
            exportName: `${this.stackName}-DomainArn`,
        });

        new cdk.CfnOutput(this, 'UserProfileName', {
            value: userProfile.userProfileName,
            description: 'Default User Profile Name',
            exportName: `${this.stackName}-UserProfileName`,
        });

        new cdk.CfnOutput(this, 'StudioUrl', {
            value: `https://${this.domainId}.studio.${this.region}.sagemaker.aws/`,
            description: 'SageMaker Studio URL',
        });
    }
}
