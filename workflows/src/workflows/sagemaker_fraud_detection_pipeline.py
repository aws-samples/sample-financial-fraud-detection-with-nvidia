import os

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.processing import Processor
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import (
    ContainerDefinition,
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
    ProductionVariant,
)
from sagemaker.core.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import CacheConfig, ProcessingStep, TrainingStep
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, OutputDataConfig


def get_session(region, default_bucket, profile_name=None):
    boto_session = (
        boto3.Session(region_name=region, profile_name=profile_name)
        if profile_name
        else boto3.Session(region_name=region)
    )
    sagemaker_session = Session(boto_session=boto_session)
    pipeline_session = PipelineSession(boto_session=boto_session)
    return sagemaker_session, pipeline_session


def register_model(
    region: str,
    model_package_group_name: str = "fraud-detection-models",
    profile_name: str = None,
    approval_status: str = "Approved",
):
    """Register a new model package version using the latest image and model artifacts."""
    session = (
        boto3.Session(region_name=region, profile_name=profile_name)
        if profile_name
        else boto3.Session(region_name=region)
    )
    sm_client = session.client("sagemaker")
    account_id = session.client("sts").get_caller_identity()["Account"]

    # Get latest model package (any status) to copy model data URL
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )

    if not response.get("ModelPackageSummaryList"):
        raise ValueError(
            f"No model packages found in group '{model_package_group_name}'."
        )

    latest_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
    latest_version = response["ModelPackageSummaryList"][0]["ModelPackageVersion"]
    print(f"Latest model package: {latest_arn} (version {latest_version})")

    existing = sm_client.describe_model_package(ModelPackageName=latest_arn)
    model_data_url = existing["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    print(f"Model data URL: {model_data_url}")

    # Register with fresh :latest image (will resolve to current digest)
    triton_image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/triton-inference-server:latest"
    )
    print(f"Using image: {triton_image_uri}")

    response = sm_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        InferenceSpecification={
            "Containers": [{"Image": triton_image_uri, "ModelDataUrl": model_data_url}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": [
                # G4dn - NVIDIA T4 (16GB)
                "ml.g4dn.xlarge",
                "ml.g4dn.2xlarge",
                "ml.g4dn.4xlarge",
                "ml.g4dn.8xlarge",
                "ml.g4dn.16xlarge",
                # G5 - NVIDIA A10G (24GB)
                "ml.g5.xlarge",
                "ml.g5.2xlarge",
                "ml.g5.4xlarge",
                "ml.g5.8xlarge",
                "ml.g5.16xlarge",
                # G6 - NVIDIA L4 (24GB)
                "ml.g6.xlarge",
                "ml.g6.2xlarge",
                "ml.g6.4xlarge",
                "ml.g6.8xlarge",
                "ml.g6.16xlarge",
                # G6e - NVIDIA L40S (48GB)
                "ml.g6e.xlarge",
                "ml.g6e.2xlarge",
                "ml.g6e.4xlarge",
                "ml.g6e.8xlarge",
                "ml.g6e.16xlarge",
                # P3 - NVIDIA V100 (16GB)
                "ml.p3.2xlarge",
            ],
        },
        ModelApprovalStatus=approval_status,
    )

    print(f"Created: {response['ModelPackageArn']}")
    return response["ModelPackageArn"]


def deploy_endpoint(
    region: str,
    role_arn: str,
    default_bucket: str,
    endpoint_name: str = "fraud-detection-endpoint",
    instance_type: str = "ml.g6e.2xlarge",
    initial_instance_count: int = 1,
    model_package_group_name: str = "fraud-detection-models",
    profile_name: str = None,
):
    """Deploy the latest approved model from the model package group."""
    sagemaker_session, _ = get_session(region, default_bucket, profile_name)
    account_id = sagemaker_session.account_id()

    # Get latest approved model package
    sm_client = boto3.client(
        "sagemaker",
        region_name=region,
    )
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )

    if not response.get("ModelPackageSummaryList"):
        raise ValueError(
            f"No approved models found in group '{model_package_group_name}'. "
            "Approve a model first in SageMaker Model Registry."
        )

    model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
    print(f"Deploying model package: {model_package_arn}")

    # Create Model from model package
    model_name = f"{endpoint_name}-model"
    try:
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(model_package_name=model_package_arn),
            execution_role_arn=role_arn,
        )
        print(f"Created model: {model_name}")
    except Exception as e:
        if "already existing" in str(e):
            print(f"Using existing model: {model_name}")
        else:
            raise

    # Create EndpointConfig
    endpoint_config_name = f"{endpoint_name}-config"
    try:
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_config_name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=model_name,
                    initial_instance_count=initial_instance_count,
                    instance_type=instance_type,
                )
            ],
        )
        print(f"Created endpoint config: {endpoint_config_name}")
    except Exception as e:
        if "already existing" in str(e):
            print(f"Using existing endpoint config: {endpoint_config_name}")
        else:
            raise

    # Create Endpoint
    try:
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_config_name,
        )
        print(f"Creating endpoint: {endpoint_name}")
    except Exception as e:
        if "already existing" in str(e):
            print(f"Using existing endpoint: {endpoint_name}")
            endpoint = Endpoint.get(endpoint_name=endpoint_name)
        else:
            raise

    endpoint.wait_for_status("InService")
    print(f"Endpoint '{endpoint_name}' is now InService.")
    return endpoint


def get_pipeline(
    region,
    role_arn,
    default_bucket,
    model_package_group_name="fraud-detection-models",
    pipeline_name="FraudDetectionPipeline",
    base_job_prefix="fraud-detect",
    profile_name=None,
):
    sagemaker_session, pipeline_session = get_session(
        region, default_bucket, profile_name
    )
    account_id = sagemaker_session.account_id()

    cache_true_config = CacheConfig(enable_caching=True, expire_after="1d")
    cache_false_config = CacheConfig(enable_caching=False, expire_after="1d")

    # Parameters
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.g6e.2xlarge"
    )
    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount", default_value=1
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.g6e.2xlarge"
    )

    input_data_url = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{default_bucket}/data/TabFormer/raw/card_transaction.v1.csv",
    )

    # GNN Hyperparameters
    gnn_hidden_channels = ParameterInteger(name="GnnHiddenChannels", default_value=32)
    gnn_n_hops = ParameterInteger(name="GnnNHops", default_value=2)
    gnn_layer = ParameterString(name="GnnLayer", default_value="SAGEConv")
    gnn_dropout_prob = ParameterFloat(name="GnnDropoutProb", default_value=0.1)
    gnn_batch_size = ParameterInteger(name="GnnBatchSize", default_value=4096)
    gnn_fan_out = ParameterInteger(name="GnnFanOut", default_value=10)
    gnn_num_epochs = ParameterInteger(name="GnnNumEpochs", default_value=8)

    # XGBoost Hyperparameters
    xgb_max_depth = ParameterInteger(name="XgbMaxDepth", default_value=6)
    xgb_learning_rate = ParameterFloat(name="XgbLearningRate", default_value=0.2)
    xgb_num_parallel_tree = ParameterInteger(name="XgbNumParallelTree", default_value=3)
    xgb_num_boost_round = ParameterInteger(name="XgbNumBoostRound", default_value=512)
    xgb_gamma = ParameterFloat(name="XgbGamma", default_value=0.0)

    # Step 1: Preprocessing (RAPIDS)
    processing_image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/rapids-preprocessing:latest"
    )

    processor = Processor(
        image_uri=processing_image_uri,
        instance_type="ml.g6e.2xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}-preprocess",
        sagemaker_session=pipeline_session,
        role=role_arn,
    )

    processor_args = processor.run(
        inputs=[
            ProcessingInput(
                input_name="input_data",
                s3_input=ProcessingS3Input(
                    s3_uri=input_data_url,
                    local_path="/opt/ml/processing/input",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="ShardedByS3Key",
                ),
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="xgb",
                s3_output=ProcessingS3Output(
                    s3_uri=f"s3://{default_bucket}/data/processed/xgb",
                    local_path="/opt/ml/processing/xgb",
                    s3_upload_mode="EndOfJob",
                ),
            ),
            ProcessingOutput(
                output_name="gnn",
                s3_output=ProcessingS3Output(
                    s3_uri=f"s3://{default_bucket}/data/processed/gnn",
                    local_path="/opt/ml/processing/gnn",
                    s3_upload_mode="EndOfJob",
                ),
            ),
        ],
    )

    step_process = ProcessingStep(
        name="PreprocessData",
        step_args=processor_args,
        cache_config=cache_true_config,
    )

    # Step 2: Training (GNN+XGBoost)
    training_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/nvidia-training-repo-sagemaker:latest"

    model_trainer = ModelTrainer(
        training_image=training_image_uri,
        compute=Compute(
            instance_type="ml.g6e.2xlarge",
            instance_count=1,
        ),
        base_job_name=f"{base_job_prefix}-train",
        sagemaker_session=pipeline_session,
        role=role_arn,
        hyperparameters={
            "gnn_hidden_channels": 32,
            "gnn_n_hops": 2,
            "gnn_layer": "SAGEConv",
            "gnn_dropout_prob": 0.1,
            "gnn_batch_size": 4096,
            "gnn_fan_out": 10,
            "gnn_num_epochs": 8,
            "xgb_max_depth": 6,
            "xgb_learning_rate": 0.2,
            "xgb_num_parallel_tree": 3,
            "xgb_num_boost_round": 512,
            "xgb_gamma": 0.0,
        },
        input_data_config=[
            InputData(
                channel_name="gnn",
                data_source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "gnn"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            InputData(
                channel_name="xgb",
                data_source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "xgb"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        ],
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{default_bucket}/model-repository"
        ),
    )

    train_args = model_trainer.train()

    step_train = TrainingStep(
        name="TrainModel",
        step_args=train_args,
        cache_config=cache_false_config,
    )

    # Step 3: Register Model
    triton_image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/triton-inference-server:latest"
    )

    model_builder = ModelBuilder(
        s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        image_uri=triton_image_uri,
        sagemaker_session=pipeline_session,
        role_arn=role_arn,
    )

    step_register = ModelStep(
        name="RegisterModel",
        step_args=model_builder.register(
            model_package_group_name=model_package_group_name,
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=[
                # G4dn - NVIDIA T4 (16GB) - Turing architecture
                "ml.g4dn.xlarge",
                "ml.g4dn.2xlarge",
                "ml.g4dn.4xlarge",
                "ml.g4dn.8xlarge",
                "ml.g4dn.16xlarge",
                # G5 - NVIDIA A10G (24GB) - Ampere architecture
                "ml.g5.xlarge",
                "ml.g5.2xlarge",
                "ml.g5.4xlarge",
                "ml.g5.8xlarge",
                "ml.g5.16xlarge",
                # G6 - NVIDIA L4 (24GB) - Ada Lovelace architecture
                "ml.g6.xlarge",
                "ml.g6.2xlarge",
                "ml.g6.4xlarge",
                "ml.g6.8xlarge",
                "ml.g6.16xlarge",
                # G6e - NVIDIA L40S (48GB) - Ada Lovelace architecture
                "ml.g6e.xlarge",
                "ml.g6e.2xlarge",
                "ml.g6e.4xlarge",
                "ml.g6e.8xlarge",
                "ml.g6e.16xlarge",
                # P3 - NVIDIA V100 (16GB) - Volta architecture
                "ml.p3.2xlarge",
            ],
            approval_status="PendingManualApproval",
        ),
    )

    # Pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            processing_instance_type,
            training_instance_count,
            training_instance_type,
            input_data_url,
            gnn_hidden_channels,
            gnn_n_hops,
            gnn_layer,
            gnn_dropout_prob,
            gnn_batch_size,
            gnn_fan_out,
            gnn_num_epochs,
            xgb_max_depth,
            xgb_learning_rate,
            xgb_num_parallel_tree,
            xgb_num_boost_round,
            xgb_gamma,
        ],
        steps=[step_process, step_train, step_register],
        sagemaker_session=pipeline_session,
    )
    return pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role-arn", required=True, help="SageMaker Execution Role ARN"
    )
    parser.add_argument("--default-bucket", required=True, help="Default S3 bucket")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile", default=None, help="AWS profile name")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Create/update pipeline")

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy endpoint")
    deploy_parser.add_argument(
        "--endpoint-name",
        default="fraud-detection-endpoint",
        help="Name for the endpoint",
    )
    deploy_parser.add_argument(
        "--instance-type",
        help="Instance type for endpoint",
    )
    deploy_parser.add_argument(
        "--model-package-group",
        default="fraud-detection-models",
        help="Model package group name",
    )

    # Register command
    register_parser = subparsers.add_parser(
        "register", help="Register new model package version"
    )
    register_parser.add_argument(
        "--model-package-group",
        default="fraud-detection-models",
        help="Model package group name",
    )
    register_parser.add_argument(
        "--approval-status",
        default="Approved",
        choices=["Approved", "PendingManualApproval", "Rejected"],
        help="Approval status for the new model package",
    )

    args = parser.parse_args()

    if args.command == "deploy":
        endpoint = deploy_endpoint(
            region=args.region,
            role_arn=args.role_arn,
            default_bucket=args.default_bucket,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            model_package_group_name=args.model_package_group,
            profile_name=args.profile,
        )
    elif args.command == "register":
        register_model(
            region=args.region,
            model_package_group_name=args.model_package_group,
            profile_name=args.profile,
            approval_status=args.approval_status,
        )
    else:
        # Default: create/update pipeline
        pipeline = get_pipeline(
            args.region, args.role_arn, args.default_bucket, profile_name=args.profile
        )
        pipeline.upsert(role_arn=args.role_arn)
        print(f"Pipeline {pipeline.name} created/updated.")
