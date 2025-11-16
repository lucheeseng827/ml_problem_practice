# Terraform Outputs for ML Practice Infrastructure

output "account_id" {
  description = "AWS Account ID"
  value       = data.aws_caller_identity.current.account_id
}

output "region" {
  description = "AWS Region"
  value       = data.aws_region.current.name
}

# SageMaker Outputs (to be uncommented when module is implemented)
# output "sagemaker_notebook_url" {
#   description = "SageMaker notebook instance URL"
#   value       = module.sagemaker.notebook_url
# }

# output "sagemaker_role_arn" {
#   description = "SageMaker execution role ARN"
#   value       = module.sagemaker.role_arn
# }

# Athena Outputs (to be uncommented when module is implemented)
# output "athena_workgroup" {
#   description = "Athena workgroup name"
#   value       = module.athena.workgroup_name
# }

# output "athena_database" {
#   description = "Athena database name"
#   value       = module.athena.database_name
# }

# Glue Outputs (to be uncommented when module is implemented)
# output "glue_catalog_database" {
#   description = "Glue catalog database name"
#   value       = module.glue.database_name
# }

# Redshift Outputs (to be uncommented when module is implemented)
# output "redshift_endpoint" {
#   description = "Redshift cluster endpoint"
#   value       = module.redshift.endpoint
#   sensitive   = true
# }

# output "redshift_cluster_id" {
#   description = "Redshift cluster ID"
#   value       = module.redshift.cluster_id
# }
