# Fraud Detection - Full Project Makefile
# ========================================
# Wraps CDK infrastructure, SageMaker pipelines, and image builds

SHELL := /bin/bash
.PHONY: help install info test test-benchmark test-real \
	cdk-synth cdk-deploy cdk-deploy-all cdk-diff cdk-destroy cdk-list \
	pipeline deploy register \
	build-triton build-training build-preprocessing build-all \
	logs clean-endpoints

# =============================================================================
# Configuration - override with environment variables
# =============================================================================
AWS_PROFILE ?= zjacobso+nvidia-Admin
AWS_REGION ?= us-east-1
MODEL_PACKAGE_GROUP ?= fraud-detection-models
ENDPOINT_NAME ?= fraud-detection-endpoint
INSTANCE_TYPE ?= ml.g6e.xlarge

# CloudFormation stack names
INFRA_STACK := SageMakerInfraStack
BLUEPRINT_STACK := NvidiaFraudDetectionBlueprint

# Directories
WORKFLOWS_DIR := workflows
INFRA_DIR := infra

# =============================================================================
# CloudFormation Lookups (cached per make invocation)
# =============================================================================
ROLE_ARN := $(shell aws cloudformation describe-stacks \
	--stack-name $(INFRA_STACK) \
	--region $(AWS_REGION) \
	--profile $(AWS_PROFILE) \
	--query 'Stacks[0].Outputs[?OutputKey==`SageMakerExecutionRoleArn`].OutputValue' \
	--output text 2>/dev/null)

BUCKET := $(shell aws cloudformation describe-stacks \
	--stack-name $(BLUEPRINT_STACK) \
	--region $(AWS_REGION) \
	--profile $(AWS_PROFILE) \
	--query 'Stacks[0].Outputs[?OutputKey==`ModelBucketName`].OutputValue' \
	--output text 2>/dev/null)

# =============================================================================
# Help
# =============================================================================
help:
	@echo "Fraud Detection - Project Commands"
	@echo "==================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install           - Install all dependencies (CDK + Python)"
	@echo "  make info              - Show current configuration"
	@echo ""
	@echo "CDK Infrastructure:"
	@echo "  make cdk-list          - List all stacks"
	@echo "  make cdk-synth         - Synthesize CloudFormation templates"
	@echo "  make cdk-diff          - Show infrastructure changes"
	@echo "  make cdk-deploy STACK=StackName  - Deploy specific stack"
	@echo "  make cdk-deploy-all    - Deploy all stacks"
	@echo "  make cdk-destroy STACK=StackName - Destroy specific stack"
	@echo ""
	@echo "SageMaker Pipeline:"
	@echo "  make pipeline          - Create/update the SageMaker pipeline"
	@echo ""
	@echo "Model Deployment:"
	@echo "  make deploy            - Deploy endpoint (latest approved model)"
	@echo "  make deploy ENDPOINT_NAME=name INSTANCE_TYPE=ml.g5.xlarge"
	@echo "  make register          - Register new model package version"
	@echo ""
	@echo "Image Building:"
	@echo "  make build-triton      - Build Triton inference image"
	@echo "  make build-training    - Build training image"
	@echo "  make build-preprocessing - Build preprocessing image"
	@echo "  make build-all         - Build all images"
	@echo ""
	@echo "Utilities:"
	@echo "  make logs              - Fetch latest endpoint logs"
	@echo "  make clean-endpoints   - Delete all endpoint configs"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run endpoint smoke tests"
	@echo "  make test-benchmark    - Run tests with latency benchmark"
	@echo "  make test-real         - Run tests with real data from the testing dataset"
	@echo ""
	@echo "Current Config: AWS_PROFILE=$(AWS_PROFILE) AWS_REGION=$(AWS_REGION)"

# =============================================================================
# Setup & Info
# =============================================================================
install:
	@echo "Installing CDK dependencies..."
	cd $(INFRA_DIR) && npm install
	@echo ""
	@echo "Installing Python dependencies..."
	cd $(WORKFLOWS_DIR) && uv sync
	@echo ""
	@echo "Done! Run 'make info' to verify configuration."

info:
	@echo "Current Configuration"
	@echo "====================="
	@echo "AWS_PROFILE:    $(AWS_PROFILE)"
	@echo "AWS_REGION:     $(AWS_REGION)"
	@echo "ROLE_ARN:       $(ROLE_ARN)"
	@echo "BUCKET:         $(BUCKET)"
	@echo ""
	@echo "Deployment Settings"
	@echo "==================="
	@echo "ENDPOINT_NAME:  $(ENDPOINT_NAME)"
	@echo "INSTANCE_TYPE:  $(INSTANCE_TYPE)"
	@echo "MODEL_GROUP:    $(MODEL_PACKAGE_GROUP)"

# =============================================================================
# CDK Infrastructure Commands
# =============================================================================
cdk-list:
	@cd $(INFRA_DIR) && npx cdk list --profile $(AWS_PROFILE)

cdk-synth:
	@cd $(INFRA_DIR) && npx cdk synth --profile $(AWS_PROFILE)

cdk-diff:
	@cd $(INFRA_DIR) && npx cdk diff --profile $(AWS_PROFILE)

cdk-deploy:
ifndef STACK
	@echo "Usage: make cdk-deploy STACK=StackName"
	@echo "Available stacks:"
	@cd $(INFRA_DIR) && npx cdk list --profile $(AWS_PROFILE)
else
	@cd $(INFRA_DIR) && npx cdk deploy $(STACK) --profile $(AWS_PROFILE) --require-approval never
endif

cdk-deploy-all:
	@echo "Deploying all stacks..."
	@cd $(INFRA_DIR) && npx cdk deploy --all --profile $(AWS_PROFILE) --require-approval never

cdk-destroy:
ifndef STACK
	@echo "Usage: make cdk-destroy STACK=StackName"
	@echo "Available stacks:"
	@cd $(INFRA_DIR) && npx cdk list --profile $(AWS_PROFILE)
else
	@cd $(INFRA_DIR) && npx cdk destroy $(STACK) --profile $(AWS_PROFILE) --force
endif

# =============================================================================
# SageMaker Pipeline Commands
# =============================================================================
pipeline:
	@echo "Creating/updating SageMaker pipeline..."
	@cd $(WORKFLOWS_DIR) && uv run python -m workflows.sagemaker_fraud_detection_pipeline \
		--role-arn "$(ROLE_ARN)" \
		--default-bucket "$(BUCKET)" \
		--region $(AWS_REGION) \
		--profile $(AWS_PROFILE) \
		pipeline

deploy:
	@echo "Deploying endpoint $(ENDPOINT_NAME) on $(INSTANCE_TYPE)..."
	@cd $(WORKFLOWS_DIR) && uv run python -m workflows.sagemaker_fraud_detection_pipeline \
		--role-arn "$(ROLE_ARN)" \
		--default-bucket "$(BUCKET)" \
		--region $(AWS_REGION) \
		--profile $(AWS_PROFILE) \
		deploy \
		--endpoint-name $(ENDPOINT_NAME) \
		--instance-type $(INSTANCE_TYPE)

register:
	@echo "Registering new model package version..."
	@cd $(WORKFLOWS_DIR) && uv run python -m workflows.sagemaker_fraud_detection_pipeline \
		--role-arn "$(ROLE_ARN)" \
		--default-bucket "$(BUCKET)" \
		--region $(AWS_REGION) \
		--profile $(AWS_PROFILE) \
		register

# =============================================================================
# Image Building (CodeBuild)
# =============================================================================
build-triton:
	@echo "Triggering Triton image build..."
	@aws codebuild start-build \
		--project-name triton-inference-image-build \
		--profile $(AWS_PROFILE) \
		--region $(AWS_REGION) \
		--query 'build.{id:id,status:buildStatus}' \
		--output table

build-training:
	@echo "Triggering training image build..."
	@aws codebuild start-build \
		--project-name sagemaker-training-image-copy \
		--profile $(AWS_PROFILE) \
		--region $(AWS_REGION) \
		--query 'build.{id:id,status:buildStatus}' \
		--output table

build-preprocessing:
	@echo "Triggering preprocessing image build..."
	@aws codebuild start-build \
		--project-name sagemaker-preprocessing-image-build \
		--profile $(AWS_PROFILE) \
		--region $(AWS_REGION) \
		--query 'build.{id:id,status:buildStatus}' \
		--output table

build-all: build-triton build-training build-preprocessing
	@echo ""
	@echo "All builds triggered. Check CodeBuild console for progress."

# =============================================================================
# Utilities
# =============================================================================
logs:
	@echo "Fetching latest endpoint logs..."
	@LOG_GROUP=$$(aws logs describe-log-groups \
		--log-group-name-prefix /aws/sagemaker/Endpoints/fraud-detection \
		--profile $(AWS_PROFILE) \
		--region $(AWS_REGION) \
		--query 'logGroups | sort_by(@, &creationTime) | [-1].logGroupName' \
		--output text 2>/dev/null); \
	if [ -z "$$LOG_GROUP" ] || [ "$$LOG_GROUP" = "None" ]; then \
		echo "No endpoint logs found."; \
		exit 0; \
	fi; \
	STREAM=$$(aws logs describe-log-streams \
		--log-group-name "$$LOG_GROUP" \
		--profile $(AWS_PROFILE) \
		--region $(AWS_REGION) \
		--order-by LastEventTime \
		--descending \
		--limit 1 \
		--query 'logStreams[0].logStreamName' \
		--output text); \
	echo "Log group: $$LOG_GROUP"; \
	echo "Log stream: $$STREAM"; \
	echo ""; \
	aws logs get-log-events \
		--log-group-name "$$LOG_GROUP" \
		--log-stream-name "$$STREAM" \
		--profile $(AWS_PROFILE) \
		--region $(AWS_REGION) \
		--limit 100 | jq -r '.events[].message'

clean-endpoints:
	@echo "Deleting all endpoint configs..."
	@aws sagemaker list-endpoint-configs \
		--profile $(AWS_PROFILE) \
		--region $(AWS_REGION) \
		--query 'EndpointConfigs[].EndpointConfigName' \
		--output text 2>/dev/null | tr '\t' '\n' | \
		xargs -I {} aws sagemaker delete-endpoint-config \
			--endpoint-config-name {} \
			--profile $(AWS_PROFILE) \
			--region $(AWS_REGION) 2>/dev/null || true
	@echo "Done."

# =============================================================================
# Testing
# =============================================================================
test:
	@cd $(WORKFLOWS_DIR) && uv run python -m workflows.test_endpoint \
		--endpoint-name $(ENDPOINT_NAME) \
		--region $(AWS_REGION) \
		--profile $(AWS_PROFILE)

test-benchmark:
	@cd $(WORKFLOWS_DIR) && uv run python -m workflows.test_endpoint \
		--endpoint-name $(ENDPOINT_NAME) \
		--region $(AWS_REGION) \
		--profile $(AWS_PROFILE) \
		--benchmark

test-real:
	@cd $(WORKFLOWS_DIR) && uv run python -m workflows.test_endpoint \
		--endpoint-name $(ENDPOINT_NAME) \
		--region $(AWS_REGION) \
		--profile $(AWS_PROFILE) \
		--use-real-data \
		--bucket $(BUCKET) \
		--max-transactions 2500
