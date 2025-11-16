# Terraform Infrastructure for ML Practice

This directory contains Terraform configurations for provisioning AWS resources for ML workloads.

## 📋 Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.5.0 installed
- Appropriate AWS permissions (Administrator or specific IAM permissions)

## 🏗️ Infrastructure Components

This Terraform configuration will provision:

### 1. **AWS SageMaker**
   - SageMaker notebook instances for interactive ML development
   - SageMaker training jobs infrastructure
   - SageMaker endpoints for model serving
   - IAM roles and policies

### 2. **AWS Athena**
   - Athena workgroup for SQL queries
   - S3 bucket for query results
   - Glue Data Catalog integration

### 3. **AWS Glue**
   - Glue databases for metadata catalog
   - Glue crawlers for schema discovery
   - Glue ETL jobs for data transformation
   - IAM roles for Glue execution

### 4. **AWS Redshift**
   - Redshift cluster for data warehousing
   - VPC and security groups
   - Parameter groups
   - IAM roles for Redshift Spectrum

## 🚀 Quick Start

### 1. Configure Variables

```bash
# Copy example variables
cp terraform.tfvars.example terraform.tfvars

# Edit with your settings
vim terraform.tfvars
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Plan Infrastructure

```bash
terraform plan
```

### 4. Apply Configuration

```bash
terraform apply
```

### 5. Destroy Infrastructure (when done)

```bash
terraform destroy
```

## 📁 Directory Structure

```
terraform/
├── main.tf                      # Main Terraform configuration
├── variables.tf                 # Input variables
├── outputs.tf                   # Output values
├── terraform.tfvars.example     # Example variable values
├── README.md                    # This file
├── sagemaker/                   # SageMaker module (to be implemented)
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── athena/                      # Athena module (to be implemented)
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── glue/                        # Glue module (to be implemented)
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── redshift/                    # Redshift module (to be implemented)
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
└── modules/                     # Shared modules
    └── common/
```

## 🔧 Configuration Options

### Environment Variables

Set these in `terraform.tfvars`:

```hcl
aws_region   = "us-east-1"
environment  = "dev"  # dev, staging, or prod
project_name = "ml-practice"
```

### SageMaker Configuration

```hcl
sagemaker_notebook_instance_type = "ml.t3.medium"  # or ml.t3.large, ml.m5.xlarge
sagemaker_training_instance_type = "ml.m5.large"   # or ml.p3.2xlarge for GPU
```

### Redshift Configuration

```hcl
redshift_cluster_type    = "single-node"  # or "multi-node"
redshift_node_type       = "dc2.large"    # or "ra3.xlplus", "dc2.8xlarge"
redshift_number_of_nodes = 1              # only for multi-node
```

## 💰 Cost Estimation

**Estimated Monthly Costs (us-east-1, dev environment):**

| Service | Resource | Monthly Cost (approx) |
|---------|----------|----------------------|
| SageMaker | ml.t3.medium notebook (8h/day) | ~$50 |
| Redshift | dc2.large single-node (24/7) | ~$180 |
| Athena | Per query (1TB scanned) | ~$5 |
| Glue | Crawler + Catalog | ~$5 |
| S3 | Storage (100GB) + requests | ~$3 |
| **Total** | | **~$243/month** |

**Cost Optimization Tips:**
- Stop SageMaker notebooks when not in use
- Use Redshift pause/resume for dev environments
- Use S3 lifecycle policies for old data
- Monitor usage with AWS Cost Explorer

## 🔒 Security Best Practices

1. **IAM Roles**: Use least-privilege IAM roles
2. **Encryption**: Enable encryption at rest and in transit
3. **VPC**: Deploy Redshift in private subnets
4. **Secrets**: Use AWS Secrets Manager for credentials
5. **Logging**: Enable CloudTrail and VPC Flow Logs

## 📊 Monitoring

After deployment, monitor resources using:

```bash
# SageMaker notebook status
aws sagemaker describe-notebook-instance --notebook-instance-name <name>

# Redshift cluster status
aws redshift describe-clusters --cluster-identifier <cluster-id>

# Athena query execution
aws athena list-query-executions --work-group <workgroup>
```

## 🔄 CI/CD Integration

Integrate Terraform with CI/CD:

```yaml
# Example GitHub Actions workflow
name: Terraform Deploy
on:
  push:
    branches: [main]
    paths: [terraform/**]

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: hashicorp/setup-terraform@v2
      - run: terraform init
      - run: terraform plan
      - run: terraform apply -auto-approve
```

## 📝 State Management

For production, use remote state:

```hcl
terraform {
  backend "s3" {
    bucket         = "your-terraform-state-bucket"
    key            = "ml-practice/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}
```

## 🚧 Current Status

⚠️ **This infrastructure is currently a placeholder.**

The following modules need to be implemented:
- [ ] SageMaker module
- [ ] Athena module
- [ ] Glue module
- [ ] Redshift module
- [ ] VPC and networking
- [ ] IAM roles and policies
- [ ] S3 buckets for data storage

## 🤝 Contributing

When implementing modules:

1. Follow Terraform best practices
2. Use variables for all configurable values
3. Add comprehensive outputs
4. Include README in each module
5. Test in dev environment first

## 📚 Resources

- [AWS SageMaker Terraform Docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/sagemaker_notebook_instance)
- [AWS Athena Terraform Docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/athena_workgroup)
- [AWS Glue Terraform Docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/glue_catalog_database)
- [AWS Redshift Terraform Docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/redshift_cluster)
