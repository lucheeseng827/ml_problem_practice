# =========================================
# SageMaker Module Outputs
# =========================================

# IAM Role
output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "sagemaker_execution_role_name" {
  description = "Name of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.name
}

# Notebook Instance
output "notebook_instance_name" {
  description = "Name of the SageMaker notebook instance"
  value       = var.create_notebook_instance ? aws_sagemaker_notebook_instance.ml_notebook[0].name : null
}

output "notebook_instance_url" {
  description = "URL of the SageMaker notebook instance"
  value       = var.create_notebook_instance ? aws_sagemaker_notebook_instance.ml_notebook[0].url : null
}

output "notebook_instance_arn" {
  description = "ARN of the SageMaker notebook instance"
  value       = var.create_notebook_instance ? aws_sagemaker_notebook_instance.ml_notebook[0].arn : null
}

# Model Registry
output "model_package_group_name" {
  description = "Name of the model package group"
  value       = var.create_model_registry ? aws_sagemaker_model_package_group.model_group[0].model_package_group_name : null
}

output "model_package_group_arn" {
  description = "ARN of the model package group"
  value       = var.create_model_registry ? aws_sagemaker_model_package_group.model_group[0].arn : null
}

# Endpoint
output "endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = var.create_endpoint ? aws_sagemaker_endpoint.endpoint[0].name : null
}

output "endpoint_arn" {
  description = "ARN of the SageMaker endpoint"
  value       = var.create_endpoint ? aws_sagemaker_endpoint.endpoint[0].arn : null
}

output "endpoint_config_name" {
  description = "Name of the endpoint configuration"
  value       = var.create_endpoint ? aws_sagemaker_endpoint_configuration.endpoint_config[0].name : null
}

# Feature Store
output "feature_group_name" {
  description = "Name of the feature group"
  value       = var.create_feature_store ? aws_sagemaker_feature_group.feature_group[0].feature_group_name : null
}

output "feature_group_arn" {
  description = "ARN of the feature group"
  value       = var.create_feature_store ? aws_sagemaker_feature_group.feature_group[0].arn : null
}

# SageMaker Project
output "sagemaker_project_id" {
  description = "ID of the SageMaker project"
  value       = var.create_sagemaker_project ? aws_sagemaker_project.mlops_project[0].id : null
}

output "sagemaker_project_arn" {
  description = "ARN of the SageMaker project"
  value       = var.create_sagemaker_project ? aws_sagemaker_project.mlops_project[0].arn : null
}

# CloudWatch
output "training_log_group_name" {
  description = "Name of CloudWatch log group for training jobs"
  value       = aws_cloudwatch_log_group.training_jobs.name
}

output "endpoint_log_group_name" {
  description = "Name of CloudWatch log group for endpoints"
  value       = aws_cloudwatch_log_group.endpoints.name
}

# Auto-scaling
output "autoscaling_target_resource_id" {
  description = "Resource ID of the auto-scaling target"
  value       = var.create_endpoint && var.enable_autoscaling ? aws_appautoscaling_target.sagemaker_target[0].resource_id : null
}

output "autoscaling_policy_name" {
  description = "Name of the auto-scaling policy"
  value       = var.create_endpoint && var.enable_autoscaling ? aws_appautoscaling_policy.sagemaker_policy[0].name : null
}

# Summary
output "summary" {
  description = "Summary of created SageMaker resources"
  value = {
    notebook_created       = var.create_notebook_instance
    model_registry_created = var.create_model_registry
    endpoint_created       = var.create_endpoint
    feature_store_created  = var.create_feature_store
    project_created        = var.create_sagemaker_project
    autoscaling_enabled    = var.create_endpoint && var.enable_autoscaling
    monitoring_enabled     = var.enable_monitoring
  }
}
