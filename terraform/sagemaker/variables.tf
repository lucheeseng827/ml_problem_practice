# =========================================
# SageMaker Module Variables
# =========================================

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# =========================================
# S3 Configuration
# =========================================

variable "s3_bucket_arn" {
  description = "ARN of S3 bucket for SageMaker data/models"
  type        = string
}

variable "s3_bucket_uri" {
  description = "URI of S3 bucket (s3://bucket-name)"
  type        = string
}

# =========================================
# Notebook Instance Configuration
# =========================================

variable "create_notebook_instance" {
  description = "Whether to create a SageMaker notebook instance"
  type        = bool
  default     = true
}

variable "notebook_instance_type" {
  description = "Instance type for SageMaker notebook"
  type        = string
  default     = "ml.t3.medium"
}

variable "notebook_volume_size" {
  description = "Size of EBS volume for notebook (GB)"
  type        = number
  default     = 20
}

variable "default_code_repository" {
  description = "Default Git repository for notebook"
  type        = string
  default     = ""
}

variable "enable_lifecycle_config" {
  description = "Enable lifecycle configuration for notebook"
  type        = bool
  default     = true
}

# =========================================
# Model Registry Configuration
# =========================================

variable "create_model_registry" {
  description = "Whether to create a model package group (registry)"
  type        = bool
  default     = true
}

# =========================================
# Endpoint Configuration
# =========================================

variable "create_endpoint" {
  description = "Whether to create a SageMaker endpoint"
  type        = bool
  default     = false  # Default false to avoid costs
}

variable "model_name" {
  description = "Name of the SageMaker model to deploy"
  type        = string
  default     = ""
}

variable "endpoint_instance_type" {
  description = "Instance type for SageMaker endpoint"
  type        = string
  default     = "ml.m5.large"
}

variable "endpoint_initial_instance_count" {
  description = "Initial number of instances for endpoint"
  type        = number
  default     = 1
}

variable "enable_serverless" {
  description = "Enable serverless inference"
  type        = bool
  default     = false
}

variable "serverless_max_concurrency" {
  description = "Max concurrent invocations for serverless"
  type        = number
  default     = 10
}

variable "serverless_memory_size" {
  description = "Memory size (MB) for serverless endpoint"
  type        = number
  default     = 2048
  validation {
    condition     = contains([1024, 2048, 3072, 4096, 5120, 6144], var.serverless_memory_size)
    error_message = "Memory size must be 1024, 2048, 3072, 4096, 5120, or 6144 MB."
  }
}

# =========================================
# Data Capture Configuration
# =========================================

variable "enable_data_capture" {
  description = "Enable data capture for model monitoring"
  type        = bool
  default     = true
}

variable "data_capture_sampling_percentage" {
  description = "Percentage of requests to capture (0-100)"
  type        = number
  default     = 100
  validation {
    condition     = var.data_capture_sampling_percentage >= 0 && var.data_capture_sampling_percentage <= 100
    error_message = "Sampling percentage must be between 0 and 100."
  }
}

# =========================================
# Feature Store Configuration
# =========================================

variable "create_feature_store" {
  description = "Whether to create a feature store"
  type        = bool
  default     = false
}

variable "feature_definitions" {
  description = "List of feature definitions for feature store"
  type = list(object({
    name = string
    type = string  # String, Integral, or Fractional
  }))
  default = [
    { name = "user_id", type = "String" },
    { name = "event_time", type = "String" }
  ]
}

variable "enable_online_store" {
  description = "Enable online feature store (for real-time inference)"
  type        = bool
  default     = true
}

variable "enable_online_store_encryption" {
  description = "Enable encryption for online store"
  type        = bool
  default     = false
}

variable "glue_database_name" {
  description = "Glue database name for offline feature store"
  type        = string
  default     = "ml_practice"
}

variable "kms_key_id" {
  description = "KMS key ID for encryption"
  type        = string
  default     = ""
}

# =========================================
# SageMaker Project Configuration
# =========================================

variable "create_sagemaker_project" {
  description = "Whether to create a SageMaker Project for CI/CD"
  type        = bool
  default     = false
}

variable "service_catalog_product_id" {
  description = "Service Catalog product ID for SageMaker project"
  type        = string
  default     = ""
}

variable "service_catalog_artifact_id" {
  description = "Service Catalog provisioning artifact ID"
  type        = string
  default     = ""
}

# =========================================
# Monitoring Configuration
# =========================================

variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring and alarms"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention period (days)"
  type        = number
  default     = 7
}

variable "alarm_sns_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarms"
  type        = string
  default     = ""
}

# =========================================
# Auto-Scaling Configuration
# =========================================

variable "enable_autoscaling" {
  description = "Enable auto-scaling for endpoint"
  type        = bool
  default     = true
}

variable "autoscaling_min_capacity" {
  description = "Minimum number of instances for auto-scaling"
  type        = number
  default     = 1
}

variable "autoscaling_max_capacity" {
  description = "Maximum number of instances for auto-scaling"
  type        = number
  default     = 3
}

variable "autoscaling_target_invocations" {
  description = "Target invocations per instance for auto-scaling"
  type        = number
  default     = 1000
}
