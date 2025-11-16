# =========================================
# S3 Module Outputs
# =========================================

output "bucket_id" {
  description = "ID of the S3 bucket"
  value       = aws_s3_bucket.ml_bucket.id
}

output "bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.ml_bucket.arn
}

output "bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.ml_bucket.bucket
}

output "bucket_domain_name" {
  description = "Domain name of the S3 bucket"
  value       = aws_s3_bucket.ml_bucket.bucket_domain_name
}

output "bucket_regional_domain_name" {
  description = "Regional domain name of the S3 bucket"
  value       = aws_s3_bucket.ml_bucket.bucket_regional_domain_name
}

output "bucket_uri" {
  description = "S3 URI of the bucket"
  value       = "s3://${aws_s3_bucket.ml_bucket.bucket}"
}

output "logging_bucket_id" {
  description = "ID of the logging bucket"
  value       = var.enable_access_logging ? aws_s3_bucket.logging_bucket[0].id : null
}

output "folders_created" {
  description = "List of folders created in the bucket"
  value       = keys(aws_s3_object.folders)
}
