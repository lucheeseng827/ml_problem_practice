import boto3

# Set the EKS cluster name and nodegroup name
cluster_name = "your-cluster-name"
nodegroup_name = "your-nodegroup-name"

# Create an EKS client
eks_client = boto3.client("eks")

# Get the ARN of the nodegroup
response = eks_client.describe_nodegroup(
    clusterName=cluster_name, nodegroupName=nodegroup_name
)
nodegroup_arn = response["nodegroup"]["nodegroupArn"]

# Create an EC2 client
ec2_client = boto3.client("ec2")

# Get the tags of the nodegroup instance(s)
response = ec2_client.describe_instances(
    Filters=[
        {
            "Name": "tag:aws:cloudformation:stack-name",
            "Values": [nodegroup_arn.split("/")[-1]],
        }
    ]
)

# Check if the nodegroup instance(s) have the required tag(s)
for reservation in response["Reservations"]:
    for instance in reservation["Instances"]:
        tags = instance["Tags"]
        required_tags = {
            "Key": "your-required-tag-key",
            "Value": "your-required-tag-value",
        }
        if required_tags not in tags:
            print(
                f"Instance {instance['InstanceId']} is not tagged with {required_tags}"
            )
