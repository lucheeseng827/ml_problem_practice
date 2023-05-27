import tecton

# Define the S3 data source
s3_data_source = tecton.S3DataSource(
    name="image_data_source",
    bucket="your-s3-bucket",
    file_format="parquet",
    file_pattern="images/*.parquet",
)

# Define the TensorFlow training component
tensorflow_trainer = tecton.TFTrainer(
    name="image_classifier_trainer",
    training_script="path/to/training_script.py",
    dependencies=["tensorflow"],
    input_data=s3_data_source,
)

# Define the pipeline
image_pipeline = tecton.Pipeline(
    name="image_processing_pipeline",
    schedule_interval="30 0 * * *",  # Run daily at midnight
    pipeline_components=[tensorflow_trainer],
)

# Register the pipeline with Tecton
tecton.register_pipeline(image_pipeline)
