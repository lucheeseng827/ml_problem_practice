import boto3
import pprint


client = boto3.client('ec2')
response = client.describe_volumes(
    Filters=[{'Name': 'tag:Name',            'Values': ['foo']
              }
             ]
)

volumes = response['Volumes']

for volume in volumes:
    print(f'Volume {volume["VolumeId"]} has state {volume["State"]}')

for volume in volumes:
    client.delete_volume(VolumeId=volume['VolumeId'])
