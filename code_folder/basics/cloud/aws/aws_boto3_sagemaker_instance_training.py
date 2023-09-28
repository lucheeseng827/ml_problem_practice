import boto3

# Initialize boto3 clients for SageMaker and Application Autoscaling
sagemaker_client = boto3.client("sagemaker")
autoscale_client = boto3.client("application-autoscaling")

# Define your SageMaker role, data sources, and other training parameters
role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
data_location = "s3://path/to/your/data"
output_location = "s3://path/to/your/output"
container_image = "container-image-arn"

training_params = {
    "AlgorithmSpecification": {
        "TrainingImage": container_image,
        "TrainingInputMode": "File",
    },
    "RoleArn": role,
    "OutputDataConfig": {"S3OutputPath": output_location},
    "ResourceConfig": {
        "InstanceType": "ml.m4.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 10,
    },
    "TrainingJobName": "MyTrainingJob",
    "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
    "InputDataConfig": [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": data_location,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }
    ],
}

# Create SageMaker training job
sagemaker_client.create_training_job(**training_params)

# Define autoscaling parameters
resource_id = f'training-job/{training_params["TrainingJobName"]}'
scalable_dimension = "sagemaker:training-job:DesiredInstanceCount"
autoscaling_params = {
    "ResourceId": resource_id,
    "ScalableDimension": scalable_dimension,
    "ServiceNamespace": "sagemaker",
    "MinCapacity": 1,
    "MaxCapacity": 2,
    "RoleARN": role,
}

# Register SageMaker training job as a scalable target
autoscale_client.register_scalable_target(**autoscaling_params)

# Define scaling policy
scaling_policy = {
    "PolicyName": "MyScalingPolicy",
    "ServiceNamespace": "sagemaker",
    "ResourceId": resource_id,
    "ScalableDimension": scalable_dimension,
    "PolicyType": "TargetTrackingScaling",
    "TargetTrackingScalingPolicyConfiguration": {
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
        },
        "ScaleOutCooldown": 60,
        "ScaleInCooldown": 300,
    },
}

# Put scaling policy to the SageMaker training job
autoscale_client.put_scaling_policy(**scaling_policy)
