import boto3
import datetime

def lambda_handler(event, context):
    # Create a Boto3 EC2 client
    ec2_client = boto3.client('ec2')

    # Get the current timestamp
    current_time = datetime.datetime.now()

    # Calculate the start and end times for the 24-hour window
    start_time = current_time - datetime.timedelta(hours=24)
    end_time = current_time

    # Retrieve the spot instances running in the specified time window
    response = ec2_client.describe_spot_instance_requests(
        Filters=[
            {
                'Name': 'state',
                'Values': ['active']
            },
            {
                'Name': 'launch-time',
                'Values': [start_time.strftime('%Y-%m-%dT%H:%M:%S')]
            }
        ]
    )

    # Calculate the total cost for the spot instances
    total_cost = 0.0
    for request in response['SpotInstanceRequests']:
        instance_type = request['LaunchSpecification']['InstanceType']
        spot_price = float(request['SpotPrice'])
        run_duration = (end_time - request['CreateTime']).total_seconds() / 3600
        cost = spot_price * run_duration
        total_cost += cost

    # Print the total cost
    print(f"Total cost for spot instances in the past 24 hours: ${total_cost:.2f}")

    # You can also return the cost if needed
    return {
        'statusCode': 200,
        'body': f"Total cost for spot instances in the past 24 hours: ${total_cost:.2f}"
    }
