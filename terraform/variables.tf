# Terraform Variables for ML Practice Infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "ml-practice"
}

# SageMaker Variables
variable "sagemaker_notebook_instance_type" {
  description = "SageMaker notebook instance type"
  type        = string
  default     = "ml.t3.medium"
}

variable "sagemaker_training_instance_type" {
  description = "SageMaker training instance type"
  type        = string
  default     = "ml.m5.large"
}

# Athena Variables
variable "athena_workgroup" {
  description = "Athena workgroup name"
  type        = string
  default     = "ml-practice-workgroup"
}

variable "athena_database" {
  description = "Athena/Glue database name"
  type        = string
  default     = "ml_practice"
}

# Redshift Variables
variable "redshift_cluster_type" {
  description = "Redshift cluster type (single-node or multi-node)"
  type        = string
  default     = "single-node"
}

variable "redshift_node_type" {
  description = "Redshift node type"
  type        = string
  default     = "dc2.large"
}

variable "redshift_number_of_nodes" {
  description = "Number of Redshift nodes (only for multi-node)"
  type        = number
  default     = 1
}

variable "redshift_master_username" {
  description = "Redshift master username"
  type        = string
  default     = "admin"
  sensitive   = true
}

variable "redshift_database_name" {
  description = "Redshift database name"
  type        = string
  default     = "ml_practice"
}

# Tagging
variable "additional_tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}
