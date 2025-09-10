import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
import * as iam from 'aws-cdk-lib/aws-iam';
import { NagSuppressions } from 'cdk-nag';
import { Construct } from 'constructs';

export interface TarExtractorStackProps extends cdk.StackProps {

  modelBucketName: string;

  modelPrefix?: string;

  ssmParameter?: string;

}
export class TarExtractorStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: TarExtractorStackProps) {
    super(scope, id, props);

    const sourceBucket = s3.Bucket.fromBucketName(this, 'ModelBucket', props.modelBucketName)

    // Destination bucket for extracted files
    const destinationBucket = new s3.Bucket(this, 'ModelRegistryBucket', {
      bucketName: props.modelBucketName + '-model-registry',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      serverAccessLogsPrefix: 'access-logs/',
      enforceSSL: true,
      encryption: s3.BucketEncryption.S3_MANAGED
    });

    // Lambda function for tar extraction
    const extractorFunction = new lambda.Function(this, 'TarExtractorFunction', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      timeout: cdk.Duration.minutes(15),
      memorySize: 4096,
      environment: {
        DESTINATION_BUCKET: destinationBucket.bucketName,
        SSM_PARAMETER_NAME: '/triton/model',
      },
      code: lambda.Code.fromInline(`
import json
import boto3
import tarfile
import tempfile
import os
from urllib.parse import unquote_plus

s3_client = boto3.client('s3')
ssm_client = boto3.client('ssm')

def clear_destination_bucket(bucket_name, prefix='model-repository/'):
    """Clear all objects in the destination bucket with the given prefix"""
    try:
        print(f"Clearing destination bucket {bucket_name} with prefix {prefix}")
        
        # List all objects with the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        objects_to_delete = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects_to_delete.append({'Key': obj['Key']})
        
        # Delete objects in batches of 1000 (S3 limit)
        if objects_to_delete:
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i:i+1000]
                s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={'Objects': batch}
                )
                print(f"Deleted {len(batch)} objects from {bucket_name}")
        else:
            print(f"No objects found to delete in {bucket_name}/{prefix}")
            
    except Exception as e:
        print(f"Error clearing destination bucket: {str(e)}")
        # Don't fail the entire operation if clearing fails
        pass

def handler(event, context):
    try:
        # Get model name from SSM parameter
        ssm_parameter_name = os.environ['SSM_PARAMETER_NAME']
        print(f"Getting model name from SSM parameter: {ssm_parameter_name}")
        
        try:
            response = ssm_client.get_parameter(Name=ssm_parameter_name)
            model_name = response['Parameter']['Value']
            print(f"Model name from SSM: {model_name}")
        except Exception as ssm_error:
            print(f"Error getting SSM parameter {ssm_parameter_name}: {str(ssm_error)}")
            raise ssm_error
        
        # Parse S3 event
        for record in event['Records']:
            source_bucket = record['s3']['bucket']['name']
            source_key = unquote_plus(record['s3']['object']['key'])
            
            print(f"Processing {source_key} from {source_bucket}")
            
            # Check if this is the model we're looking for
            expected_key = f"output/{model_name}/output/model.tar.gz"
            if source_key != expected_key:
                print(f"Skipping {source_key} - not the expected model file {expected_key}")
                continue
            
            # Skip if not a tar.gz file
            if not source_key.lower().endswith(('.tar.gz', '.tgz')):
                print(f"Skipping {source_key} - not a tar.gz file")
                continue
            
            # Clear destination bucket before processing new model
            destination_bucket = os.environ['DESTINATION_BUCKET']
            clear_destination_bucket(destination_bucket)
            
            # Download tar.gz file to temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                local_tar_path = os.path.join(temp_dir, 'archive.tar.gz')
                
                print(f"Downloading {source_key} to {local_tar_path}")
                s3_client.download_file(source_bucket, source_key, local_tar_path)
                
                # Extract tar.gz file
                extract_dir = os.path.join(temp_dir, 'extracted')
                os.makedirs(extract_dir, exist_ok=True)
                
                print(f"Extracting {local_tar_path}")
                with tarfile.open(local_tar_path, 'r:gz') as tar:
                    # Security check - prevent path traversal
                    def is_safe_path(path):
                        return not (path.startswith('/') or '..' in path)
                    
                    safe_members = [m for m in tar.getmembers() if is_safe_path(m.name)]
                    tar.extractall(path=extract_dir, members=safe_members)
                
                # Process extracted files with specific model repository structure
                
                # The bash script extracts and uses python_backend_model_repository
                # We need to copy the python_backend_model_repository contents to model-repository
                python_backend_path = os.path.join(extract_dir, 'python_backend_model_repository')
                
                if os.path.exists(python_backend_path):
                    print(f"Found python_backend_model_repository at {python_backend_path}")
                    
                    # Walk through the python_backend_model_repo directory
                    for root, dirs, files in os.walk(python_backend_path):
                        for file in files:
                            local_file_path = os.path.join(root, file)
                            # Get relative path from python_backend_model_repo
                            relative_path = os.path.relpath(local_file_path, python_backend_path)
                            
                            # Normalize path separators for cross-platform compatibility
                            relative_path = relative_path.replace('\\\\\\\\', '/')
                            
                            # Upload to model-repository structure
                            destination_key = f"model-repository/{relative_path}"
                            
                            print(f"Uploading {local_file_path} to s3://{destination_bucket}/{destination_key}")
                            
                            # Upload file to S3
                            try:
                                s3_client.upload_file(local_file_path, destination_bucket, destination_key)
                                print(f"Successfully uploaded {destination_key}")
                            except Exception as upload_error:
                                print(f"Error uploading {destination_key}: {str(upload_error)}")
                                raise upload_error
                else:
                    print(f"python_backend_model_repo not found at {python_backend_path}")
                    
                    # Fallback to original mapping logic if structure is different
                    model_mappings = {
                        'model/model_repository/model': 'model-repository/model-other',
                        'model/model_repository/xgboost': 'model-repository/xgboost', 
                        'model/python_backend_model_repository/prediction_and_shapely': 'model-repository/models'
                    }
                    
                    # Walk through extracted files and upload with proper structure
                    for root, dirs, files in os.walk(extract_dir):
                        for file in files:
                            local_file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_file_path, extract_dir)
                            
                            # Normalize path separators for cross-platform compatibility
                            relative_path = relative_path.replace('\\\\\\\\', '/')
                            
                            # Find matching model mapping
                            destination_key = None
                            for source_pattern, dest_pattern in model_mappings.items():
                                if relative_path.startswith(source_pattern + '/'):
                                    # Replace the source pattern with destination pattern
                                    file_within_model = relative_path[len(source_pattern + '/'):]
                                    destination_key = f"{dest_pattern}/{file_within_model}"
                                    break
                            
                            # Skip files that don't match our expected structure
                            if destination_key is None:
                                print(f"Skipping {relative_path} - doesn't match expected model structure")
                                continue
                            
                            print(f"Uploading {local_file_path} to s3://{destination_bucket}/{destination_key}")
                            
                            # Upload file to S3
                            try:
                                s3_client.upload_file(local_file_path, destination_bucket, destination_key)
                                print(f"Successfully uploaded {destination_key}")
                            except Exception as upload_error:
                                print(f"Error uploading {destination_key}: {str(upload_error)}")
                                raise upload_error
                
                print(f"Successfully processed {source_key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Successfully processed tar.gz files')
        }
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        raise e
      `),
    });

    // Grant permissions to read from source bucket
    sourceBucket.grantRead(extractorFunction);

    // Grant permissions to write to destination bucket
    destinationBucket.grantReadWrite(extractorFunction);

    // Grant permissions to delete objects from destination bucket (for rewriting)
    destinationBucket.grantDelete(extractorFunction);


    // Grant permissions to read SSM parameter
    extractorFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['ssm:GetParameter'],
      resources: [`arn:aws:ssm:${this.region}:${this.account}:parameter/triton/model`],
    }));

    // Add S3 event notification to trigger Lambda for files in output/ prefix
    sourceBucket.addEventNotification(
      s3.EventType.OBJECT_CREATED,
      new s3n.LambdaDestination(extractorFunction),
      {
        prefix: 'output/',
        suffix: '.tar.gz'
      }
    );

    // Also trigger on .tgz files in output/ prefix
    sourceBucket.addEventNotification(
      s3.EventType.OBJECT_CREATED,
      new s3n.LambdaDestination(extractorFunction),
      {
        prefix: 'output/',
        suffix: '.tgz'
      }
    );

    // Add CDK Nag suppressions
    NagSuppressions.addResourceSuppressions(
      destinationBucket,
      [
        {
          id: 'AwsSolutions-S1',
          reason: 'Server access logs are enabled with serverAccessLogsPrefix'
        },
        {
          id: 'AwsSolutions-S10',
          reason: 'SSL enforcement is enabled with enforceSSL: true'
        }
      ]
    );

    NagSuppressions.addResourceSuppressions(
      extractorFunction,
      [
        {
          id: 'AwsSolutions-IAM4',
          reason: 'AWS managed policy required for Lambda basic execution',
          appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole']
        },
        {
          id: 'AwsSolutions-IAM5',
          reason: 'Wildcard permissions required for S3 bucket operations',
          appliesTo: [
            'Action::s3:GetBucket*',
            'Action::s3:GetObject*',
            'Action::s3:List*',
            'Action::s3:Abort*',
            'Action::s3:DeleteObject*',
            'Resource::*',
            'Resource::arn:aws:s3:::ml-on-containers/*',
            'Resource::<ModelRegistryBucketBC3B3633.Arn>/*'
          ]
        },
        {
          id: 'AwsSolutions-L1',
          reason: 'Using Python 3.12 which is the latest supported runtime'
        }
      ],
      true
    );

    // Suppress bucket notifications handler warnings
    NagSuppressions.addResourceSuppressionsByPath(
      this,
      '/NvidiaFraudDetectionBlueprintModelExtractor/BucketNotificationsHandler050a0587b7544547bf325f094a3db834/Role',
      [
        {
          id: 'AwsSolutions-IAM4',
          reason: 'AWS managed policy required for S3 bucket notifications',
          appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole']
        }
      ]
    );

    // Suppress bucket notifications handler default policy warnings
    NagSuppressions.addResourceSuppressionsByPath(
      this,
      '/NvidiaFraudDetectionBlueprintModelExtractor/BucketNotificationsHandler050a0587b7544547bf325f094a3db834/Role/DefaultPolicy',
      [
        {
          id: 'AwsSolutions-IAM5',
          reason: 'Wildcard permissions required for S3 bucket notifications functionality to manage bucket configurations',
          appliesTo: ['Resource::*']
        }
      ]
    );

    // Outputs
    new cdk.CfnOutput(this, 'SourceBucketName', {
      value: sourceBucket.bucketName,
      description: 'Name of the source S3 bucket for tar.gz files',
    });

    new cdk.CfnOutput(this, 'DestinationBucketName', {
      value: destinationBucket.bucketName,
      description: 'Name of the destination S3 bucket for extracted files',
    });

    new cdk.CfnOutput(this, 'LambdaFunctionName', {
      value: extractorFunction.functionName,
      description: 'Name of the Lambda function',
    });
  }
}
