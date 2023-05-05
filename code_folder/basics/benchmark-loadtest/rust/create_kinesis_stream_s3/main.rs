use rusoto_core::Region;
use rusoto_kinesis::{Kinesis, KinesisClient, PutRecordInput, PutRecordOutput};
use rusoto_s3::{GetObjectRequest, S3Client, S3};
use std::io::Read;

#[tokio::main]
async fn main() {
    let s3_client = S3Client::new(Region::UsEast1);
    let kinesis_client = KinesisClient::new(Region::UsEast1);

    // Set up the S3 object and bucket name
    let object_key = "example.txt";
    let bucket_name = "my-s3-bucket";

    // Get object from S3
    let get_req = GetObjectRequest {
        bucket: bucket_name.to_string(),
        key: object_key.to_string(),
        ..Default::default()
    };

    let mut s3_object = match s3_client.get_object(get_req).await {
        Ok(obj) => obj,
        Err(e) => panic!("Error getting S3 object: {:?}", e),
    };

    // Read data from S3 object
    let mut s3_data = String::new();
    match s3_object.body.take() {
        Some(mut b) => {
            match b.read_to_string(&mut s3_data) {
                Ok(_) => (),
                Err(e) => panic!("Error reading S3 data: {:?}", e),
            }
        }
        None => panic!("Error reading S3 object body"),
    }

    // Set up the Kinesis stream name and partition key
    let stream_name = "my-kinesis-stream";
    let partition_key = "partition-1";

    // Put data into Kinesis stream
    let put_req = PutRecordInput {
        data: s3_data.into_bytes(),
        stream_name: stream_name.to_string(),
        partition_key: partition_key.to_string(),
        ..Default::default()
    };

    match kinesis_client.put_record(put_req).await {
        Ok(resp) => {
            println!("Data successfully put into Kinesis stream. Shard ID: {}", resp.shard_id);
        }
        Err(e) => panic!("Error putting data into Kinesis stream: {:?}", e),
    }
}
