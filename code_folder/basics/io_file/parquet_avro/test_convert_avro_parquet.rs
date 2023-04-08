#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_s3_object_to_parquet_and_avro_files() {
        let bucket_name = "my-bucket";
        let object_key = "data.txt";
        let parquet_output = "data.parquet";
        let avro_output = "data.avro";

        let access_key = "my-access-key";
        let secret_key = "my-secret-key";
        let region = Region::UsEast1;

        let provider = StaticProvider::new(access_key, secret_key, None, None);
        let s3_client = S3Client::new_with(
            rusoto_core::HttpClient::new().unwrap(),
            provider,
            region,
        );

        // Upload sample data file to S3 bucket
        let sample_data = b"1\tAlice\t100.0\n2\tBob\t200.0\n3\tCharlie\t300.0\n";
        let put_obj_req = PutObjectRequest {
            bucket: bucket_name.to_string(),
            key: object_key.to_string(),
            body: Some(sample_data.to_vec().into()),
            ..Default::default()
        };
        match s3_client.put_object(put_obj_req).sync() {
            Ok(_) => {}
            Err(error) => {
                panic!("Failed to put object to S3: {:?}", error);
            }
        }

        // Convert S3 object to Parquet file
        let parquet_data = convert_s3_object_to_parquet(&s3_client, bucket_name, object_key, parquet_output);
        assert!(parquet_data.is_ok());

        // Convert S3 object to Avro file
        let avro_data = convert_s3_object_to_avro(&s3_client, bucket_name, object_key, avro_output);
        assert!(avro_data.is_ok());

        // Clean up output files
        match std::fs::remove_file(parquet_output) {
            Ok(_) => {}
            Err(_) => {}
        }
        match std::fs::remove_file(avro_output) {
            Ok(_) => {}
            Err(_) => {}
        }
    }
}
