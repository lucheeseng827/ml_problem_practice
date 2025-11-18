# =========================================
# S3 Module Variables
# =========================================

variable "bucket_name" {
  description = "Name of the S3 bucket"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]*[a-z0-9]$", var.bucket_name))
    error_message = "Bucket name must start and end with lowercase letter or number, and contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Versioning
variable "enable_versioning" {
  description = "Enable versioning for the bucket"
  type        = bool
  default     = true
}

# Encryption
variable "kms_key_id" {
  description = "KMS key ID for encryption (empty for AES256)"
  type        = string
  default     = ""
}

# Lifecycle
variable "enable_lifecycle_rules" {
  description = "Enable lifecycle rules"
  type        = bool
  default     = true
}

variable "archive_after_days" {
  description = "Days before archiving training data to Glacier"
  type        = number
  default     = 90
}

variable "model_archive_after_days" {
  description = "Days before archiving models to Glacier IR"
  type        = number
  default     = 180
}

variable "expire_after_days" {
  description = "Days before deleting archived data"
  type        = number
  default     = 365
}

# Logging
variable "enable_access_logging" {
  description = "Enable S3 access logging"
  type        = bool
  default     = false
}

# Notifications
variable "enable_notifications" {
  description = "Enable S3 event notifications"
  type        = bool
  default     = false
}

variable "lambda_function_arns" {
  description = "List of Lambda function ARNs for S3 notifications"
  type        = list(string)
  default     = []
}
