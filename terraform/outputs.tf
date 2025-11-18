# Terraform Outputs for ML Practice Infrastructure

# =========================================
# General Outputs
# =========================================

output "account_id" {
  description = "AWS Account ID"
  value       = data.aws_caller_identity.current.account_id
}

output "region" {
  description = "AWS Region"
  value       = data.aws_region.current.name
}

# =========================================
# S3 Outputs
# =========================================

output "ml_bucket_name" {
  description = "Name of the ML S3 bucket"
  value       = module.ml_s3.bucket_name
}

output "ml_bucket_arn" {
  description = "ARN of the ML S3 bucket"
  value       = module.ml_s3.bucket_arn
}

output "ml_bucket_uri" {
  description = "S3 URI of the ML bucket"
  value       = module.ml_s3.bucket_uri
}

# =========================================
# SageMaker Outputs
# =========================================

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = module.sagemaker.sagemaker_execution_role_arn
}

output "sagemaker_notebook_name" {
  description = "Name of the SageMaker notebook instance"
  value       = module.sagemaker.notebook_instance_name
}

output "sagemaker_notebook_url" {
  description = "URL of the SageMaker notebook instance"
  value       = module.sagemaker.notebook_instance_url
}

output "sagemaker_model_registry_name" {
  description = "Name of the SageMaker model package group"
  value       = module.sagemaker.model_package_group_name
}

output "sagemaker_endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = module.sagemaker.endpoint_name
}

output "sagemaker_feature_group_name" {
  description = "Name of the SageMaker feature group"
  value       = module.sagemaker.feature_group_name
}

# =========================================
# Summary Output
# =========================================

output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    s3_bucket              = module.ml_s3.bucket_name
    sagemaker_notebook     = module.sagemaker.notebook_instance_name != null ? "Created" : "Not created"
    sagemaker_model_registry = module.sagemaker.model_package_group_name != null ? "Created" : "Not created"
    sagemaker_endpoint     = module.sagemaker.endpoint_name != null ? "Created" : "Not created"
    feature_store          = module.sagemaker.feature_group_name != null ? "Created" : "Not created"
  }
}

# =========================================
# Athena Outputs (Placeholder)
# =========================================
# output "athena_workgroup" {
#   description = "Athena workgroup name"
#   value       = module.athena.workgroup_name
# }

# output "athena_database" {
#   description = "Athena database name"
#   value       = module.athena.database_name
# }

# =========================================
# Glue Outputs (Placeholder)
# =========================================
# output "glue_catalog_database" {
#   description = "Glue catalog database name"
#   value       = module.glue.database_name
# }

# =========================================
# Redshift Outputs (Placeholder)
# =========================================
# output "redshift_endpoint" {
#   description = "Redshift cluster endpoint"
#   value       = module.redshift.endpoint
#   sensitive   = true
# }

# output "redshift_cluster_id" {
#   description = "Redshift cluster ID"
#   value       = module.redshift.cluster_id
# }
