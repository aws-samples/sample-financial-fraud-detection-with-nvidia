import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

export interface SageMakerDomainStackProps extends cdk.StackProps {
    readonly domainName?: string;
    readonly executionRoleArn: string;
    /** GitHub repo URL to clone notebooks from on first boot. */
    readonly notebookRepoUrl?: string;
    /** Branch to checkout after cloning. */
    readonly notebookRepoBranch?: string;
}

export class SageMakerDomainStack extends cdk.Stack {
    public readonly domain: sagemaker.CfnDomain;
    public readonly domainId: string;
    public readonly vpc: ec2.Vpc;

    constructor(scope: Construct, id: string, props: SageMakerDomainStackProps) {
        super(scope, id, props);

        const domainName = props.domainName || 'fraud-detection-domain';

        // Create a VPC for SageMaker Studio (required)
        this.vpc = new ec2.Vpc(this, 'SageMakerVPC', {
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
        const subnetIds = this.vpc.privateSubnets.map(subnet => subnet.subnetId);

        // --- Lifecycle config: clone notebooks from GitHub on first boot ---
        const repoUrl = props.notebookRepoUrl
            ?? 'https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia.git';
        const repoBranch = props.notebookRepoBranch ?? 'main';

        const onCreateScript = [
            '#!/bin/bash',
            'set -eux',
            '',
            'REPO_DIR="/home/sagemaker-user/fraud-detection"',
            'if [ ! -d "$REPO_DIR" ]; then',
            `  git clone --depth 1 --branch ${repoBranch} ${repoUrl} "$REPO_DIR"`,
            'fi',
            '',
            '# Install notebook dependencies if requirements.txt exists',
            'if [ -f "$REPO_DIR/notebooks/requirements.txt" ]; then',
            '  pip install --quiet -r "$REPO_DIR/notebooks/requirements.txt"',
            'fi',
        ].join('\n');

        const lifecycleConfig = new sagemaker.CfnStudioLifecycleConfig(this, 'NotebookLifecycleConfig', {
            studioLifecycleConfigAppType: 'JupyterServer',
            studioLifecycleConfigName: 'clone-notebooks-on-create',
            studioLifecycleConfigContent: cdk.Fn.base64(onCreateScript),
        });

        // --- SageMaker Domain ---
        this.domain = new sagemaker.CfnDomain(this, 'SageMakerDomain', {
            authMode: 'IAM',
            defaultUserSettings: {
                executionRole: props.executionRoleArn,
                securityGroups: [this.vpc.vpcDefaultSecurityGroup],
                jupyterServerAppSettings: {
                    defaultResourceSpec: {
                        instanceType: 'system',
                        lifecycleConfigArn: lifecycleConfig.attrStudioLifecycleConfigArn,
                    },
                    lifecycleConfigArns: [lifecycleConfig.attrStudioLifecycleConfigArn],
                },
            },
            domainName: domainName,
            subnetIds: subnetIds,
            vpcId: this.vpc.vpcId,
            appNetworkAccessType: 'PublicInternetOnly',
        });

        this.domainId = this.domain.attrDomainId;

        const userProfileName = 'default-user';

        // Create a default user profile
        const userProfile = new sagemaker.CfnUserProfile(this, 'DefaultUserProfile', {
            domainId: this.domainId,
            userProfileName: userProfileName,
            userSettings: {
                executionRole: props.executionRoleArn,
            },
        });

        userProfile.addDependency(this.domain);

        // --- JupyterServer app: auto-provisions a notebook instance on deploy ---
        const jupyterApp = new sagemaker.CfnApp(this, 'JupyterServerApp', {
            appName: 'default',
            appType: 'JupyterServer',
            domainId: this.domainId,
            userProfileName: userProfileName,
        });

        jupyterApp.addDependency(userProfile);

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
