import boto3
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel


session = boto3.Session(
    aws_access_key_id='YOUR_ACCESS_KEY',
    aws_secret_access_key='YOUR_SECRET_KEY',
    region_name='YOUR_REGION'
)

role = get_execution_role()
sagemaker_client = session.client('sagemaker')

processing_input = ProcessingInput(
    input_name='input_data',
    source='s3://your-bucket/input-data/',
    destination='/opt/ml/processing/input'
)

processing_output = ProcessingOutput(
    output_name='preprocessed_data',
    source='/opt/ml/processing/output/preprocessed',
    destination='s3://your-bucket/preprocessed-data/'
)

processor = SKLearnProcessor(
    framework_version='0.20.0',
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1
)

model_output_location = 's3://your-bucket/model-output/'
model = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.xlarge',
    framework_version='0.20.0',
    output_path=model_output_location
)

pipeline_model = PipelineModel(
    name='model-pipeline',
    role=role,
    models=[
        model,
    ]
)

notebook_instance_name = 'my-notebook-instance'
sagemaker_client.create_notebook_instance(
    NotebookInstanceName=notebook_instance_name,
    InstanceType='ml.t2.medium',
    RoleArn=role,
    Tags=[
        {
            'Key': 'Name',
            'Value': 'My Notebook'
        }
    ]
)

pipeline_name = 'my-pipeline'
pipeline_description = 'My Model Pipeline'
pipeline_instance_count = 1
pipeline_instance_type = 'ml.m5.xlarge'
pipeline_execution_role = role

sagemaker_client.create_pipeline(
    PipelineName=pipeline_name,
    PipelineDefinition=pipeline_model.definition(),
    PipelineDescription=pipeline_description,
    RoleArn=pipeline_execution_role
)
