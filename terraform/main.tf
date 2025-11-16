# ML Practice Infrastructure - Terraform Main Configuration
# This file orchestrates the provisioning of AWS resources for ML workloads

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to use S3 backend for state management
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "ml-practice/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "terraform-state-lock"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "ML-Practice"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Local variables
locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name

  common_tags = {
    Project     = "ML-Practice"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# =========================================
# S3 Module - ML Data and Artifacts Storage
# =========================================

module "ml_s3" {
  source = "./modules/s3"

  bucket_name              = "${var.project_name}-ml-${local.account_id}"
  environment              = var.environment
  tags                     = local.common_tags
  enable_versioning        = true
  enable_lifecycle_rules   = var.enable_lifecycle_rules
  archive_after_days       = var.s3_archive_after_days
  model_archive_after_days = var.s3_model_archive_after_days
  expire_after_days        = var.s3_expire_after_days
  enable_access_logging    = var.enable_s3_access_logging
}

# =========================================
# SageMaker Module - ML Training and Deployment
# =========================================

module "sagemaker" {
  source = "./sagemaker"

  project_name = var.project_name
  tags         = local.common_tags

  # S3 configuration
  s3_bucket_arn = module.ml_s3.bucket_arn
  s3_bucket_uri = module.ml_s3.bucket_uri

  # Notebook configuration
  create_notebook_instance = var.create_sagemaker_notebook
  notebook_instance_type    = var.sagemaker_notebook_instance_type
  enable_lifecycle_config   = true

  # Model registry
  create_model_registry = var.create_model_registry

  # Endpoint configuration (disabled by default to avoid costs)
  create_endpoint                 = var.create_sagemaker_endpoint
  endpoint_instance_type          = var.sagemaker_endpoint_instance_type
  endpoint_initial_instance_count = var.sagemaker_endpoint_initial_instance_count
  enable_serverless               = var.sagemaker_enable_serverless

  # Data capture
  enable_data_capture              = var.enable_data_capture
  data_capture_sampling_percentage = var.data_capture_sampling_percentage

  # Feature store
  create_feature_store = var.create_feature_store
  glue_database_name   = var.athena_database

  # Monitoring
  enable_monitoring    = var.enable_sagemaker_monitoring
  log_retention_days   = var.log_retention_days
  alarm_sns_topic_arn  = ""  # Add SNS topic if needed

  # Auto-scaling
  enable_autoscaling            = var.enable_sagemaker_autoscaling
  autoscaling_min_capacity      = var.sagemaker_autoscaling_min_capacity
  autoscaling_max_capacity      = var.sagemaker_autoscaling_max_capacity
  autoscaling_target_invocations = var.sagemaker_autoscaling_target_invocations
}

# =========================================
# Athena Module (Placeholder - to be implemented)
# =========================================
# module "athena" {
#   source = "./athena"
#
#   environment        = var.environment
#   project_name       = var.project_name
#   s3_bucket_uri      = module.ml_s3.bucket_uri
#   database_name      = var.athena_database
#   workgroup_name     = var.athena_workgroup
# }

# =========================================
# Glue Module (Placeholder - to be implemented)
# =========================================
# module "glue" {
#   source = "./glue"
#
#   environment        = var.environment
#   project_name       = var.project_name
#   s3_bucket_arn      = module.ml_s3.bucket_arn
#   database_name      = var.athena_database
# }

# =========================================
# Redshift Module (Placeholder - to be implemented)
# =========================================
# module "redshift" {
#   source = "./redshift"
#
#   environment          = var.environment
#   project_name         = var.project_name
#   cluster_type         = var.redshift_cluster_type
#   node_type            = var.redshift_node_type
#   number_of_nodes      = var.redshift_number_of_nodes
#   master_username      = var.redshift_master_username
#   database_name        = var.redshift_database_name
# }
