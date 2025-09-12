# Makefile for AWS ECR setup and Nvidia container deployment
# This Makefile handles setting up ECR repositories and pushing Nvidia containers for
# the fraud detection blueprint project

# AWS Configuration
AWS_REGION ?= us-west-2
AWS_ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text)

# Container Registry Configuration
ECR_REPO_NAME := nvidia-fraud-detection
NVIDIA_BASE_IMAGE := nvcr.io/nvidia/cugraph/financial-fraud-training:1.0.0
ECR_IMAGE_URI := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO_NAME):latest

# Build Configuration
DOCKERFILE_DIR := ./container

# Colors for terminal output
YELLOW := \033[0;33m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help check-deps create-ecr pull-nvidia-image tag-image push-image setup-ecr clean

# Default target
help:
	@echo "$(YELLOW)Financial Fraud Detection - AWS Container Setup$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  help               : Display this help message"
	@echo "  check-deps         : Check if required dependencies are installed"
	@echo "  create-ecr         : Create ECR repository if it doesn't exist"
	@echo "  pull-nvidia-image  : Pull Nvidia base image from NGC"
	@echo "  tag-image          : Tag Nvidia image for ECR"
	@echo "  push-image         : Push tagged image to ECR"
	@echo "  setup-ecr          : Complete ECR setup process (all steps)"
	@echo "  clean              : Clean up local Docker images"
	@echo ""
	@echo "$(YELLOW)Example:$(NC)"
	@echo "  make setup-ecr AWS_REGION=us-east-1"

# Check if required tools are installed
check-deps:
	@echo "$(YELLOW)Checking dependencies...$(NC)"
	@which aws > /dev/null || (echo "$(RED)AWS CLI not found. Please install it.$(NC)" && exit 1)
	@which docker > /dev/null || (echo "$(RED)Docker not found. Please install it.$(NC)" && exit 1)
	@aws sts get-caller-identity > /dev/null || (echo "$(RED)AWS credentials not configured. Please run 'aws configure'.$(NC)" && exit 1)
	@echo "$(GREEN)All dependencies are installed.$(NC)"

# Create ECR repository if it doesn't exist
create-ecr: check-deps
	@echo "$(YELLOW)Creating ECR repository if it doesn't exist...$(NC)"
	@if ! aws ecr describe-repositories --repository-names $(ECR_REPO_NAME) --region $(AWS_REGION) > /dev/null 2>&1; then \
		echo "Creating ECR repository: $(ECR_REPO_NAME)"; \
		aws ecr create-repository --repository-name $(ECR_REPO_NAME) --region $(AWS_REGION); \
		echo "$(GREEN)ECR repository created: $(ECR_REPO_NAME)$(NC)"; \
	else \
		echo "$(GREEN)ECR repository already exists: $(ECR_REPO_NAME)$(NC)"; \
	fi

# Login to ECR
ecr-login: check-deps
	@echo "$(YELLOW)Logging into ECR...$(NC)"
	@aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	@echo "$(GREEN)Successfully logged into ECR.$(NC)"

# Pull Nvidia base image from NGC
pull-nvidia-image: check-deps
	@echo "$(YELLOW)Pulling Nvidia base image from NGC...$(NC)"
	@echo "This may take some time depending on your internet connection."
	@if ! docker pull $(NVIDIA_BASE_IMAGE); then \
		echo "$(RED)Failed to pull Nvidia image. Make sure you're logged in to NGC registry.$(NC)"; \
		echo "Run 'docker login nvcr.io' with your NGC API key as password and '\$$oauthtoken' as username."; \
		exit 1; \
	fi
	@echo "$(GREEN)Successfully pulled Nvidia base image.$(NC)"

# Tag the image for ECR
tag-image: pull-nvidia-image
	@echo "$(YELLOW)Tagging image for ECR...$(NC)"
	@docker tag $(NVIDIA_BASE_IMAGE) $(ECR_IMAGE_URI)
	@echo "$(GREEN)Successfully tagged image: $(ECR_IMAGE_URI)$(NC)"

# Push the image to ECR
push-image: create-ecr ecr-login tag-image
	@echo "$(YELLOW)Pushing image to ECR...$(NC)"
	@docker push $(ECR_IMAGE_URI)
	@echo "$(GREEN)Successfully pushed image to ECR: $(ECR_IMAGE_URI)$(NC)"

# Build custom container (if needed)
build-custom:
	@if [ -d "$(DOCKERFILE_DIR)" ] && [ -f "$(DOCKERFILE_DIR)/Dockerfile" ]; then \
		echo "$(YELLOW)Building custom container...$(NC)"; \
		docker build -t $(ECR_REPO_NAME):latest $(DOCKERFILE_DIR); \
		docker tag $(ECR_REPO_NAME):latest $(ECR_IMAGE_URI); \
		echo "$(GREEN)Successfully built custom container.$(NC)"; \
	else \
		echo "$(YELLOW)No Dockerfile found in $(DOCKERFILE_DIR). Skipping custom build.$(NC)"; \
	fi

# Complete ECR setup process
setup-ecr: check-deps create-ecr pull-nvidia-image tag-image push-image
	@echo "$(GREEN)=========================================================$(NC)"
	@echo "$(GREEN)ECR setup complete!$(NC)"
	@echo "$(GREEN)Container is now available at: $(ECR_IMAGE_URI)$(NC)"
	@echo "$(GREEN)Use this URI in your SageMaker training job.$(NC)"
	@echo "$(GREEN)=========================================================$(NC)"

# Clean up local Docker images
clean:
	@echo "$(YELLOW)Cleaning up local Docker images...$(NC)"
	-docker rmi $(ECR_IMAGE_URI) 2>/dev/null || true
	-docker rmi $(NVIDIA_BASE_IMAGE) 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete.$(NC)"
