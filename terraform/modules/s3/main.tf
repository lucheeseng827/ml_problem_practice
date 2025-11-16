# =========================================
# S3 Module for ML Data and Artifacts
# =========================================

locals {
  common_tags = merge(
    var.tags,
    {
      Module = "S3"
    }
  )
}

# =========================================
# ML Data and Artifacts Bucket
# =========================================

resource "aws_s3_bucket" "ml_bucket" {
  bucket = var.bucket_name

  tags = merge(
    local.common_tags,
    {
      Name        = var.bucket_name
      Purpose     = "ML data, models, and artifacts"
      Environment = var.environment
    }
  )
}

# Bucket versioning
resource "aws_s3_bucket_versioning" "ml_bucket_versioning" {
  bucket = aws_s3_bucket.ml_bucket.id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Disabled"
  }
}

# Server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "ml_bucket_encryption" {
  bucket = aws_s3_bucket.ml_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.kms_key_id != "" ? "aws:kms" : "AES256"
      kms_master_key_id = var.kms_key_id != "" ? var.kms_key_id : null
    }
    bucket_key_enabled = var.kms_key_id != "" ? true : false
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "ml_bucket_public_access_block" {
  bucket = aws_s3_bucket.ml_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle rules
resource "aws_s3_bucket_lifecycle_configuration" "ml_bucket_lifecycle" {
  count = var.enable_lifecycle_rules ? 1 : 0

  bucket = aws_s3_bucket.ml_bucket.id

  # Archive old training data
  rule {
    id     = "archive-training-data"
    status = "Enabled"

    filter {
      prefix = "training-data/"
    }

    transition {
      days          = var.archive_after_days
      storage_class = "GLACIER"
    }

    expiration {
      days = var.expire_after_days
    }
  }

  # Archive old model artifacts
  rule {
    id     = "archive-old-models"
    status = "Enabled"

    filter {
      prefix = "models/"
    }

    transition {
      days          = var.model_archive_after_days
      storage_class = "GLACIER_IR"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  # Delete temporary/scratch data
  rule {
    id     = "cleanup-temp-data"
    status = "Enabled"

    filter {
      prefix = "temp/"
    }

    expiration {
      days = 7
    }
  }

  # Abort incomplete multipart uploads
  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Bucket policy for SageMaker/Lambda access
resource "aws_s3_bucket_policy" "ml_bucket_policy" {
  bucket = aws_s3_bucket.ml_bucket.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DenyInsecureTransport"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.ml_bucket.arn,
          "${aws_s3_bucket.ml_bucket.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

# S3 bucket notification (for Lambda triggers)
resource "aws_s3_bucket_notification" "ml_bucket_notification" {
  count = var.enable_notifications ? 1 : 0

  bucket = aws_s3_bucket.ml_bucket.id

  # Lambda function notifications
  dynamic "lambda_function" {
    for_each = var.lambda_function_arns
    content {
      lambda_function_arn = lambda_function.value
      events              = ["s3:ObjectCreated:*"]
      filter_prefix       = "raw-data/"
    }
  }
}

# =========================================
# Logging Bucket (for access logs)
# =========================================

resource "aws_s3_bucket" "logging_bucket" {
  count = var.enable_access_logging ? 1 : 0

  bucket = "${var.bucket_name}-logs"

  tags = merge(
    local.common_tags,
    {
      Name    = "${var.bucket_name}-logs"
      Purpose = "S3 access logs"
    }
  )
}

resource "aws_s3_bucket_public_access_block" "logging_bucket_public_access_block" {
  count = var.enable_access_logging ? 1 : 0

  bucket = aws_s3_bucket.logging_bucket[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_acl" "logging_bucket_acl" {
  count = var.enable_access_logging ? 1 : 0

  bucket = aws_s3_bucket.logging_bucket[0].id
  acl    = "log-delivery-write"
}

# Enable logging on main bucket
resource "aws_s3_bucket_logging" "ml_bucket_logging" {
  count = var.enable_access_logging ? 1 : 0

  bucket = aws_s3_bucket.ml_bucket.id

  target_bucket = aws_s3_bucket.logging_bucket[0].id
  target_prefix = "access-logs/"
}

# =========================================
# Create folder structure with objects
# =========================================

resource "aws_s3_object" "folders" {
  for_each = toset([
    "raw-data/",
    "processed-data/",
    "training-data/",
    "validation-data/",
    "test-data/",
    "models/",
    "model-artifacts/",
    "feature-store/",
    "data-capture/",
    "experiments/",
    "notebooks/",
    "scripts/",
    "temp/"
  ])

  bucket       = aws_s3_bucket.ml_bucket.id
  key          = each.value
  content_type = "application/x-directory"

  tags = local.common_tags
}
