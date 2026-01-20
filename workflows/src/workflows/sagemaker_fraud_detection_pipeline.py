import os

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.processing import Processor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
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
from sagemaker.train.configs import Compute, InputData


def get_session(region, default_bucket, profile_name=None):
    boto_session = (
        boto3.Session(region_name=region, profile_name=profile_name)
        if profile_name
        else boto3.Session(region_name=region)
    )
    sagemaker_session = Session(boto_session=boto_session)
    pipeline_session = PipelineSession(boto_session=boto_session)
    return sagemaker_session, pipeline_session


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

    # Cache config for pipeline steps
    cache_true_config = CacheConfig(enable_caching=True, expire_after="1d")
    cache_false_config = CacheConfig(enable_caching=False, expire_after="1d")

    # ========================================================================
    # Parameters
    # ========================================================================
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

    # S3 paths
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

    # ========================================================================
    # Step 1: Preprocessing (RAPIDS)
    # ========================================================================
    # Image URI for the custom RAPIDS container
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

    # Note: The preprocessing container has the script built-in as entrypoint
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

    # ========================================================================
    # Step 2: Training (GNN+XGBoost)
    # ========================================================================
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
    )

    train_args = model_trainer.train()

    step_train = TrainingStep(
        name="TrainModel",
        step_args=train_args,
        cache_config=cache_false_config,
    )

    # ========================================================================
    # Step 3: Register Model
    # ========================================================================
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
            inference_instances=["ml.g6e.2xlarge"],
            approval_status="PendingManualApproval",
        ),
    )

    # ========================================================================
    # Pipeline
    # ========================================================================
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
    args = parser.parse_args()

    pipeline = get_pipeline(
        args.region, args.role_arn, args.default_bucket, profile_name=args.profile
    )
    pipeline.upsert(role_arn=args.role_arn)
    print(f"Pipeline {pipeline.name} created/updated.")
