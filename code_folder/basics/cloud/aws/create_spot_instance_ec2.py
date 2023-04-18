import boto3

# Create an EC2 client
ec2 = boto3.client("ec2")

# Define the spot instance request parameters
spot_price = "0.05"
instance_count = 1
instance_type = "t2.micro"
availability_zone = "us-west-2a"
# Replace this with the appropriate AMI ID for your region
image_id = "ami-0c55b159cbfafe1f0"

# Create the spot instance request
response = ec2.request_spot_instances(
    DryRun=False,
    SpotPrice=spot_price,
    InstanceCount=instance_count,
    Type="one-time",
    LaunchSpecification={
        "ImageId": image_id,
        "InstanceType": instance_type,
        "Placement": {"AvailabilityZone": availability_zone},
        # Uncomment the lines below and set the appropriate values for key pair and security group
        # 'KeyName': 'your-key-pair',
        # 'SecurityGroupIds': ['your-security-group-id'],
    },
)

# Print the spot instance request ID
spot_request_id = response["SpotInstanceRequests"][0]["SpotInstanceRequestId"]
print("Spot Instance Request ID:", spot_request_id)
