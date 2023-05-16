use rusoto_core::Region;
use rusoto_ec2::{DescribeInstancesRequest, Ec2, Ec2Client};

#[tokio::main]
async fn main() {
    let client = Ec2Client::new(Region::default());

    let request = DescribeInstancesRequest::default();

    match client.describe_instances(request).await {
        Ok(response) => {
            if let Some(reservations) = response.reservations {
                for reservation in reservations {
                    if let Some(instances) = reservation.instances {
                        for instance in instances {
                            if let Some(instance_id) = instance.instance_id {
                                println!("Instance ID: {}", instance_id);
                            }
                            if let Some(state) = instance.state {
                                if let Some(name) = instance.tags
                                    .iter()
                                    .find(|tag| tag.key == Some("Name".to_string()))
                                    .and_then(|tag| tag.value.clone())
                                {
                                    println!("Name: {}", name);
                                }
                                println!("State: {:?}", state.name);
                            }
                            println!("---");
                        }
                    }
                }
            }
        }
        Err(err) => {
            eprintln!("Error listing instances: {:?}", err);
        }
    }
}
