import boto3


def lambda_handler(event, context):
    ecs = boto3.client("ecs")

    # Configure the ECS task parameters
    cluster = "your-ecs-cluster-name"
    task_definition = "your-task-definition-name"
    count = 1

    # Start the ECS task
    response = ecs.run_task(
        cluster=cluster, taskDefinition=task_definition, count=count
    )

    # Check for any failures
    failures = response.get("failures", [])
    if failures:
        print("Failed to start ECS task:")
        for failure in failures:
            print(failure)
    else:
        print("ECS task started successfully.")

    return {"statusCode": 200, "body": "ECS task started."}
