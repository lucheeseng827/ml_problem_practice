import boto3
import json
import os

def lambda_handler(event, context):
    # Get environment variables
    ecs_cluster_name = os.environ['ECS_CLUSTER_NAME']
    ecs_task_role_arn = os.environ['ECS_TASK_ROLE_ARN']
    ecs_task_execution_role_arn = os.environ['ECS_TASK_EXECUTION_ROLE_ARN']
    dockerhub_username = os.environ['DOCKERHUB_USERNAME']
    dockerhub_password = os.environ['DOCKERHUB_PASSWORD']
    golden_image_name = os.environ['GOLDEN_IMAGE_NAME']
    golden_image_version = os.environ['GOLDEN_IMAGE_VERSION']

    # Create ECS client
    ecs_client = boto3.client('ecs')

    # Define task definition for building golden image
    task_definition = {
        'family': golden_image_name,
        'taskRoleArn': ecs_task_role_arn,
        'executionRoleArn': ecs_task_execution_role_arn,
        'containerDefinitions': [
            {
                'name': 'golden-image-builder',
                'image': 'docker:latest',
                'command': [
                    'docker', 'build', '-t', f'{dockerhub_username}/{golden_image_name}:{golden_image_version}', '.'
                ],
                'environment': [
                    {
                        'name': 'DOCKERHUB_USERNAME',
                        'value': dockerhub_username
                    },
                    {
                        'name': 'DOCKERHUB_PASSWORD',
                        'value': dockerhub_password
                    }
                ],
                'workingDirectory': '/build',
                'mountPoints': [
                    {
                        'sourceVolume': 'build',
                        'containerPath': '/build',
                        'readOnly': False
                    }
                ]
            }
        ],
        'volumes': [
            {
                'name': 'build',
                'host': {
                    'sourcePath': '/tmp/build'
                }
            }
        ]
    }

    # Register task definition
    response = ecs_client.register_task_definition(
        family=task_definition['family'],
        taskRoleArn=task_definition['taskRoleArn'],
        executionRoleArn=task_definition['executionRoleArn'],
        containerDefinitions=task_definition['containerDefinitions'],
        volumes=task_definition['volumes']
    )

    # Run task to build golden image
    task_response = ecs_client.run_task(
        cluster=ecs_cluster_name,
        taskDefinition=f'{golden_image_name}:1',
        count=1,
        launchType='FARGATE',
        platformVersion='LATEST',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': ['subnet-12345678'],
                'securityGroups': ['sg-12345678'],
                'assignPublicIp': 'DISABLED'
            }
        }
    )

    # Check if task ran successfully
    if task_response['failures']:
        print(f'Task failed: {task_response["failures"]}')
    else:
        print(f'Task ran successfully: {task_response["tasks"]}')

    # Tag and push golden image to Docker Hub
    os.system(f'docker tag {dockerhub_username}/{golden_image_name}:{golden_image_version} {dockerhub_username}/{golden_image_name}:latest')
    os.system(f'docker push {dockerhub
