# =========================================
# AWS SageMaker Infrastructure Module
# =========================================
# This module provisions SageMaker resources for ML workloads

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name

  common_tags = merge(
    var.tags,
    {
      Module = "SageMaker"
    }
  )
}

# =========================================
# SageMaker Execution Role
# =========================================

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.project_name}-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# Attach AWS managed policies
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Custom policy for S3 access
resource "aws_iam_role_policy" "sagemaker_s3_policy" {
  name = "${var.project_name}-sagemaker-s3-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          var.s3_bucket_arn,
          "${var.s3_bucket_arn}/*"
        ]
      }
    ]
  })
}

# ECR access for custom containers
resource "aws_iam_role_policy" "sagemaker_ecr_policy" {
  name = "${var.project_name}-sagemaker-ecr-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# =========================================
# SageMaker Notebook Instance
# =========================================

resource "aws_sagemaker_notebook_instance" "ml_notebook" {
  count = var.create_notebook_instance ? 1 : 0

  name                    = "${var.project_name}-notebook"
  instance_type           = var.notebook_instance_type
  role_arn                = aws_iam_role.sagemaker_execution_role.arn
  platform_identifier     = "notebook-al2-v2"
  volume_size             = var.notebook_volume_size
  default_code_repository = var.default_code_repository

  lifecycle_config_name = var.enable_lifecycle_config ? aws_sagemaker_notebook_instance_lifecycle_configuration.config[0].name : null

  tags = merge(
    local.common_tags,
    {
      Name = "${var.project_name}-notebook"
    }
  )
}

# Lifecycle configuration for notebook
resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "config" {
  count = var.enable_lifecycle_config ? 1 : 0

  name = "${var.project_name}-lifecycle-config"

  on_start = base64encode(<<-EOF
    #!/bin/bash
    set -e

    # Install additional packages
    sudo -u ec2-user -i <<'EOFUSER'
    source /home/ec2-user/anaconda3/bin/activate python3
    pip install --upgrade pip
    pip install duckdb mlflow bentoml great-expectations
    conda deactivate
    EOFUSER

    echo "Lifecycle configuration completed"
  EOF
  )

  on_create = base64encode(<<-EOF
    #!/bin/bash
    set -e

    # Clone repositories or download data
    sudo -u ec2-user -i <<'EOFUSER'
    cd /home/ec2-user/SageMaker
    # git clone your-repository-url
    EOFUSER
  EOF
  )
}

# =========================================
# SageMaker Model Registry
# =========================================

resource "aws_sagemaker_model_package_group" "model_group" {
  count = var.create_model_registry ? 1 : 0

  model_package_group_name = "${var.project_name}-models"
  model_package_group_description = "Model registry for ${var.project_name}"

  tags = local.common_tags
}

# =========================================
# SageMaker Endpoint Configuration (Example)
# =========================================

resource "aws_sagemaker_endpoint_configuration" "endpoint_config" {
  count = var.create_endpoint ? 1 : 0

  name = "${var.project_name}-endpoint-config"

  production_variants {
    variant_name           = "primary"
    model_name             = var.model_name  # Must be created separately
    instance_type          = var.endpoint_instance_type
    initial_instance_count = var.endpoint_initial_instance_count
    initial_variant_weight = 1

    # Auto-scaling target
    dynamic "serverless_config" {
      for_each = var.enable_serverless ? [1] : []
      content {
        max_concurrency   = var.serverless_max_concurrency
        memory_size_in_mb = var.serverless_memory_size
      }
    }
  }

  data_capture_config {
    enable_capture              = var.enable_data_capture
    initial_sampling_percentage = var.data_capture_sampling_percentage
    destination_s3_uri          = "${var.s3_bucket_uri}/data-capture"

    capture_options {
      capture_mode = "InputAndOutput"
    }

    capture_content_type_header {
      csv_content_types  = ["text/csv"]
      json_content_types = ["application/json"]
    }
  }

  tags = local.common_tags

  lifecycle {
    create_before_destroy = true
  }
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "endpoint" {
  count = var.create_endpoint ? 1 : 0

  name                 = "${var.project_name}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.endpoint_config[0].name

  tags = local.common_tags
}

# =========================================
# SageMaker Feature Store (Feature Group)
# =========================================

resource "aws_sagemaker_feature_group" "feature_group" {
  count = var.create_feature_store ? 1 : 0

  feature_group_name = "${var.project_name}-features"
  record_identifier_feature_name = "user_id"
  event_time_feature_name        = "event_time"
  role_arn                       = aws_iam_role.sagemaker_execution_role.arn

  # Define features
  dynamic "feature_definition" {
    for_each = var.feature_definitions
    content {
      feature_name = feature_definition.value.name
      feature_type = feature_definition.value.type
    }
  }

  # Online store configuration
  online_store_config {
    enable_online_store = var.enable_online_store

    dynamic "security_config" {
      for_each = var.enable_online_store_encryption ? [1] : []
      content {
        kms_key_id = var.kms_key_id
      }
    }
  }

  # Offline store configuration
  offline_store_config {
    s3_storage_config {
      s3_uri     = "${var.s3_bucket_uri}/feature-store"
      kms_key_id = var.kms_key_id
    }

    data_catalog_config {
      table_name    = "${var.project_name}_features"
      catalog       = "AwsDataCatalog"
      database_name = var.glue_database_name
    }
  }

  tags = local.common_tags
}

# =========================================
# SageMaker Project (for CI/CD)
# =========================================

resource "aws_sagemaker_project" "mlops_project" {
  count = var.create_sagemaker_project ? 1 : 0

  project_name = "${var.project_name}-mlops"

  service_catalog_provisioning_details {
    product_id           = var.service_catalog_product_id
    provisioning_artifact_id = var.service_catalog_artifact_id
  }

  tags = local.common_tags
}

# =========================================
# CloudWatch Log Groups
# =========================================

resource "aws_cloudwatch_log_group" "training_jobs" {
  name              = "/aws/sagemaker/TrainingJobs/${var.project_name}"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "endpoints" {
  name              = "/aws/sagemaker/Endpoints/${var.project_name}"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# =========================================
# CloudWatch Alarms for Endpoint Monitoring
# =========================================

resource "aws_cloudwatch_metric_alarm" "endpoint_invocation_errors" {
  count = var.create_endpoint && var.enable_monitoring ? 1 : 0

  alarm_name          = "${var.project_name}-endpoint-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ModelInvocation4XXErrors"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "This metric monitors SageMaker endpoint 4XX errors"
  alarm_actions       = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.endpoint[0].name
    VariantName  = "primary"
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "endpoint_latency" {
  count = var.create_endpoint && var.enable_monitoring ? 1 : 0

  alarm_name          = "${var.project_name}-endpoint-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Average"
  threshold           = 1000  # 1 second
  alarm_description   = "This metric monitors SageMaker endpoint latency"
  alarm_actions       = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.endpoint[0].name
    VariantName  = "primary"
  }

  tags = local.common_tags
}

# =========================================
# Auto Scaling for Endpoint
# =========================================

resource "aws_appautoscaling_target" "sagemaker_target" {
  count = var.create_endpoint && var.enable_autoscaling ? 1 : 0

  max_capacity       = var.autoscaling_max_capacity
  min_capacity       = var.autoscaling_min_capacity
  resource_id        = "endpoint/${aws_sagemaker_endpoint.endpoint[0].name}/variant/primary"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "sagemaker_policy" {
  count = var.create_endpoint && var.enable_autoscaling ? 1 : 0

  name               = "${var.project_name}-scaling-policy"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker_target[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker_target[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
    target_value       = var.autoscaling_target_invocations
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}
