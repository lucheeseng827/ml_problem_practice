import boto3
from botocore.exceptions import ClientError

session = boto3.Session(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    region_name="YOUR_REGION",
)

eks_client = session.client("eks")


instance_types = [
    "g4dn.xlarge",
    "g4dn.2xlarge",
]  # Replace with your desired GPU instance types
min_size = 1
max_size = 5
desired_size = 3
spot = {"enabled": True}
spot["maxPrice"] = "0.20"
volume_size = 40


try:
    response = eks_client.create_nodegroup(
        clusterName="YOUR_CLUSTER_NAME",
        nodegroupName="YOUR_NODEGROUP_NAME",
        nodeRole="arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_NODEGROUP_ROLE",
        subnets=["subnet-12345678", "subnet-87654321"],  # Replace with your subnet IDs
        instanceTypes=instance_types,
        minSize=min_size,
        maxSize=max_size,
        desiredSize=desired_size,
        capacityType="SPOT",
        spotOptions=spot,
    )
    print("Node group created successfully:", response)
except ClientError as e:
    print("Error creating node group:", e)
