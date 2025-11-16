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

# =====================================
# S3 Variables
# =====================================

variable "enable_lifecycle_rules" {
  description = "Enable S3 lifecycle rules"
  type        = bool
  default     = true
}

variable "s3_archive_after_days" {
  description = "Days before archiving training data to Glacier"
  type        = number
  default     = 90
}

variable "s3_model_archive_after_days" {
  description = "Days before archiving models to Glacier IR"
  type        = number
  default     = 180
}

variable "s3_expire_after_days" {
  description = "Days before deleting archived data"
  type        = number
  default     = 365
}

variable "enable_s3_access_logging" {
  description = "Enable S3 access logging"
  type        = bool
  default     = false
}

# =====================================
# SageMaker Variables
# =====================================

variable "create_sagemaker_notebook" {
  description = "Create SageMaker notebook instance"
  type        = bool
  default     = false  # Default false to avoid costs
}

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

variable "create_model_registry" {
  description = "Create SageMaker model registry"
  type        = bool
  default     = true
}

variable "create_sagemaker_endpoint" {
  description = "Create SageMaker endpoint"
  type        = bool
  default     = false  # Default false to avoid costs
}

variable "sagemaker_endpoint_instance_type" {
  description = "SageMaker endpoint instance type"
  type        = string
  default     = "ml.m5.large"
}

variable "sagemaker_endpoint_initial_instance_count" {
  description = "Initial instance count for SageMaker endpoint"
  type        = number
  default     = 1
}

variable "sagemaker_enable_serverless" {
  description = "Enable serverless inference"
  type        = bool
  default     = false
}

variable "enable_data_capture" {
  description = "Enable data capture for model monitoring"
  type        = bool
  default     = true
}

variable "data_capture_sampling_percentage" {
  description = "Percentage of requests to capture (0-100)"
  type        = number
  default     = 100
}

variable "create_feature_store" {
  description = "Create SageMaker feature store"
  type        = bool
  default     = false
}

variable "enable_sagemaker_monitoring" {
  description = "Enable CloudWatch monitoring for SageMaker"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention period (days)"
  type        = number
  default     = 7
}

variable "enable_sagemaker_autoscaling" {
  description = "Enable auto-scaling for SageMaker endpoint"
  type        = bool
  default     = true
}

variable "sagemaker_autoscaling_min_capacity" {
  description = "Minimum instance count for auto-scaling"
  type        = number
  default     = 1
}

variable "sagemaker_autoscaling_max_capacity" {
  description = "Maximum instance count for auto-scaling"
  type        = number
  default     = 3
}

variable "sagemaker_autoscaling_target_invocations" {
  description = "Target invocations per instance"
  type        = number
  default     = 1000
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
