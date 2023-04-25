import boto3
from datetime import datetime, timedelta

# Set up AWS client
client = boto3.client('ec2')
ec2_resource = boto3.resource('ec2')
ebs_resource = boto3.resource('ec2')
eks_client = boto3.client('eks')

# Set up search criteria
project_name = 'my_project'
owner_name = 'john_doe'
start_time = datetime.utcnow() - timedelta(days=1)

# Search EC2 instances
instances = ec2_resource.instances.filter(Filters=[{'Name': 'tag:Project', 'Values': [project_name]},
                                                   {'Name': 'tag:Owner', 'Values': [owner_name]},
                                                   {'Name': 'launch-time', 'Values': [start_time.isoformat()]}])

for instance in instances:
    instance_id = instance.id
    instance_tags = instance.tags
    # Check tags and take appropriate action

# Search EBS volumes
volumes = ebs_resource.volumes.filter(Filters=[{'Name': 'tag:Project', 'Values': [project_name]},
                                                   {'Name': 'tag:Owner', 'Values': [owner_name]},
                                                   {'Name': 'create-time', 'Values': [start_time.isoformat()]}])

for volume in volumes:
    volume_id = volume.id
    volume_tags = volume.tags
    # Check tags and take appropriate action

# Search EKS clusters
clusters = eks_client.list_clusters()

for cluster in clusters['clusters']:
    cluster_name = cluster
    response = eks_client.describe_cluster(name=cluster_name)
    cluster_tags = response['tags']
    # Check tags and take appropriate action
