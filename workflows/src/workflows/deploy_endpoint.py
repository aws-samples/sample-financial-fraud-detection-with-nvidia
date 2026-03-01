#!/usr/bin/env python3
"""
Deploy SageMaker Triton Endpoint for Fraud Detection

This script deploys a SageMaker endpoint using the trained model and Triton inference container.
"""

import argparse
import sys
import time

import boto3


def get_latest_model_data(s3_client, bucket, prefix="model-repository"):
    """Find the latest model.tar.gz in the S3 bucket."""
    print(f"Searching for model artifacts in s3://{bucket}/{prefix}/...")
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    model_files = []
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            if obj['Key'].endswith('model.tar.gz'):
                model_files.append({
                    'key': obj['Key'],
                    'last_modified': obj['LastModified']
                })
    
    if not model_files:
        raise ValueError(f"No model.tar.gz found in s3://{bucket}/{prefix}/")
    
    # Sort by last modified and get the most recent
    model_files.sort(key=lambda x: x['last_modified'], reverse=True)
    latest = model_files[0]
    
    model_data_url = f"s3://{bucket}/{latest['key']}"
    print(f"Found latest model: {model_data_url}")
    print(f"Last modified: {latest['last_modified']}")
    
    return model_data_url


def get_triton_image_uri(ecr_client, repository_name="triton-inference-server", tag="latest"):
    """Get the Triton image URI from ECR."""
    try:
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        repo_uri = response['repositories'][0]['repositoryUri']
        image_uri = f"{repo_uri}:{tag}"
        print(f"Triton image: {image_uri}")
        return image_uri
    except ecr_client.exceptions.RepositoryNotFoundException:
        raise ValueError(f"ECR repository '{repository_name}' not found")


def get_execution_role(cf_client, stack_name="SageMakerInfraStack"):
    """Get SageMaker execution role ARN from CloudFormation stack."""
    try:
        response = cf_client.describe_stacks(StackName=stack_name)
        outputs = {o['OutputKey']: o['OutputValue'] for o in response['Stacks'][0]['Outputs']}
        role_arn = outputs.get('SageMakerExecutionRoleArn')
        if not role_arn:
            raise ValueError(f"SageMakerExecutionRoleArn not found in stack {stack_name}")
        print(f"Execution role: {role_arn}")
        return role_arn
    except cf_client.exceptions.ClientError:
        raise ValueError(f"CloudFormation stack '{stack_name}' not found")


def create_model(sm_client, model_name, image_uri, model_data_url, execution_role_arn):
    """Create SageMaker Model."""
    print(f"\nCreating SageMaker Model: {model_name}")
    
    try:
        sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'serve',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
                }
            },
            ExecutionRoleArn=execution_role_arn
        )
        print(f"✓ Model created: {model_name}")
        return model_name
    except sm_client.exceptions.ResourceInUse:
        print(f"Model '{model_name}' already exists, using existing model")
        return model_name


def create_endpoint_config(sm_client, config_name, model_name, instance_type, initial_instance_count=1, ami_version=None):
    """Create SageMaker Endpoint Configuration."""
    print(f"\nCreating Endpoint Configuration: {config_name}")
    print(f"  Instance type: {instance_type}")
    print(f"  Initial instance count: {initial_instance_count}")
    if ami_version:
        print(f"  AMI version: {ami_version}")
    
    production_variant = {
        'VariantName': 'AllTraffic',
        'ModelName': model_name,
        'InstanceType': instance_type,
        'InitialInstanceCount': initial_instance_count,
    }
    
    # Add AMI version if specified
    if ami_version:
        production_variant['InferenceAmiVersion'] = ami_version
    
    try:
        sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[production_variant]
        )
        print(f"✓ Endpoint config created: {config_name}")
        return config_name
    except sm_client.exceptions.ResourceInUse:
        print(f"Endpoint config '{config_name}' already exists, using existing config")
        return config_name


def create_endpoint(sm_client, endpoint_name, config_name):
    """Create SageMaker Endpoint."""
    print(f"\nCreating Endpoint: {endpoint_name}")
    
    try:
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"✓ Endpoint creation initiated: {endpoint_name}")
        return endpoint_name
    except sm_client.exceptions.ResourceInUse:
        print(f"Endpoint '{endpoint_name}' already exists, updating instead...")
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"✓ Endpoint update initiated: {endpoint_name}")
        return endpoint_name


def wait_for_endpoint(sm_client, endpoint_name, timeout_minutes=30):
    """Wait for endpoint to be InService."""
    print(f"\nWaiting for endpoint to be InService (timeout: {timeout_minutes} minutes)...")
    print("This typically takes 5-10 minutes for GPU instances.")
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"\n✗ Timeout after {timeout_minutes} minutes")
            return False
        
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        
        elapsed_min = int(elapsed / 60)
        elapsed_sec = int(elapsed % 60)
        print(f"  [{elapsed_min:02d}:{elapsed_sec:02d}] Status: {status}", end='\r')
        
        if status == 'InService':
            print(f"\n✓ Endpoint is InService after {elapsed_min}m {elapsed_sec}s")
            return True
        elif status == 'Failed':
            failure_reason = response.get('FailureReason', 'Unknown')
            print(f"\n✗ Endpoint creation failed: {failure_reason}")
            return False
        
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(
        description="Deploy SageMaker Triton endpoint for fraud detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Deploy with defaults
  %(prog)s --instance-type ml.g5.2xlarge      # Use larger instance
  %(prog)s --endpoint-name my-endpoint        # Custom endpoint name
  %(prog)s --no-wait                          # Don't wait for InService
  %(prog)s --ami-version ami-version-2025-01-07  # Use specific AMI version
""",
    )
    parser.add_argument(
        '--endpoint-name',
        default='fraud-detection-endpoint',
        help='SageMaker endpoint name (default: fraud-detection-endpoint)'
    )
    parser.add_argument(
        '--instance-type',
        default='ml.g5.xlarge',
        help='SageMaker instance type (default: ml.g5.xlarge)'
    )
    parser.add_argument(
        '--instance-count',
        type=int,
        default=1,
        help='Initial instance count (default: 1)'
    )
    parser.add_argument(
        '--ami-version',
        default='ami-version-2025-01-07',
        help='SageMaker inference AMI version (default: ami-version-2025-01-07 with NVIDIA driver 550.144.01, CUDA 12.4)'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--profile',
        default=None,
        help='AWS profile name'
    )
    parser.add_argument(
        '--bucket',
        default=None,
        help='S3 bucket with model artifacts (auto-detected if not specified)'
    )
    parser.add_argument(
        '--model-data-url',
        default=None,
        help='Direct S3 URL to model.tar.gz (overrides bucket search)'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        help='Do not wait for endpoint to be InService'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout in minutes for endpoint creation (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Setup AWS clients
    session_kwargs = {'region_name': args.region}
    if args.profile:
        session_kwargs['profile_name'] = args.profile
    
    session = boto3.Session(**session_kwargs)
    sm_client = session.client('sagemaker')
    s3_client = session.client('s3')
    ecr_client = session.client('ecr')
    cf_client = session.client('cloudformation')
    sts_client = session.client('sts')
    
    print("=" * 70)
    print("SageMaker Triton Endpoint Deployment")
    print("=" * 70)
    print(f"\nEndpoint name: {args.endpoint_name}")
    print(f"Instance type: {args.instance_type}")
    print(f"Instance count: {args.instance_count}")
    print(f"AMI version: {args.ami_version}")
    print(f"Region: {args.region}")
    print(f"Profile: {args.profile or 'default'}")
    
    try:
        # Get account ID
        account_id = sts_client.get_caller_identity()['Account']
        
        # Get bucket if not specified
        bucket = args.bucket
        if not bucket:
            bucket = f"fraud-detection-{account_id}-sm"
            print(f"\nUsing bucket: {bucket}")
        
        # Get model data URL
        if args.model_data_url:
            model_data_url = args.model_data_url
            print(f"\nUsing specified model: {model_data_url}")
        else:
            model_data_url = get_latest_model_data(s3_client, bucket)
        
        # Get Triton image URI
        image_uri = get_triton_image_uri(ecr_client)
        
        # Get execution role
        execution_role_arn = get_execution_role(cf_client)
        
        # Generate resource names
        timestamp = int(time.time())
        model_name = f"{args.endpoint_name}-model-{timestamp}"
        config_name = f"{args.endpoint_name}-config-{timestamp}"
        
        # Create resources
        create_model(sm_client, model_name, image_uri, model_data_url, execution_role_arn)
        create_endpoint_config(sm_client, config_name, model_name, args.instance_type, args.instance_count, args.ami_version)
        create_endpoint(sm_client, args.endpoint_name, config_name)
        
        # Wait for endpoint
        if not args.no_wait:
            success = wait_for_endpoint(sm_client, args.endpoint_name, args.timeout)
            if success:
                print("\n" + "=" * 70)
                print("Deployment Successful!")
                print("=" * 70)
                print(f"\nEndpoint name: {args.endpoint_name}")
                print(f"Region: {args.region}")
                print(f"\nTest the endpoint:")
                print(f"  python test_endpoint.py --endpoint-name {args.endpoint_name}")
                if args.profile:
                    print(f"  --profile {args.profile}")
                print("\n" + "=" * 70)
                return 0
            else:
                print("\n✗ Deployment failed")
                return 1
        else:
            print("\n" + "=" * 70)
            print("Endpoint creation initiated (not waiting for InService)")
            print("=" * 70)
            print(f"\nCheck status with:")
            print(f"  aws sagemaker describe-endpoint --endpoint-name {args.endpoint_name}")
            if args.profile:
                print(f"  --profile {args.profile}")
            print("\n" + "=" * 70)
            return 0
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
