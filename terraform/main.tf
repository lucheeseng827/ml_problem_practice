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

# SageMaker Module (Placeholder - to be implemented)
# module "sagemaker" {
#   source = "./sagemaker"
#
#   environment = var.environment
#   project_name = var.project_name
# }

# Athena Module (Placeholder - to be implemented)
# module "athena" {
#   source = "./athena"
#
#   environment = var.environment
#   project_name = var.project_name
# }

# Glue Module (Placeholder - to be implemented)
# module "glue" {
#   source = "./glue"
#
#   environment = var.environment
#   project_name = var.project_name
# }

# Redshift Module (Placeholder - to be implemented)
# module "redshift" {
#   source = "./redshift"
#
#   environment = var.environment
#   project_name = var.project_name
# }
