from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
from azure.mgmt.datafactory.models import *
from azure.synapse.accesscontrol import AccessControlClient
from azure.synapse.artifacts import ArtifactsClient

# Set up the Azure Synapse Access Control client
credential = DefaultAzureCredential()
access_control_client = AccessControlClient(
    endpoint="https://<Your-Synapse-Workspace>.dev.azuresynapse.net",
    credential=credential,
)

# Assuming you've set up Synapse SQL to query blob data, run your query
sql_query = """
    -- Your SQL query to run on Blob data
    SELECT * FROM OPENROWSET(data) limit 2000
"""
# Execute the query (this is a simplification, you'll usually want error handling and pagination logic here)
response = access_control_client.role_assignments.create_role_assignment(
    scope="/", role_assignment_name="..."
)

# Set up Azure Data Factory client
adf_client = DataFactoryManagementClient(credential, "<Your-Subscription-Id>")

# Assuming you have a pipeline set up in Azure Data Factory, trigger it
pipeline_run = adf_client.pipelines.create_run(
    "<Your-Resource-Group-Name>", "<Your-Data-Factory-Name>", "<Your-Pipeline-Name>", {}
)

# Output the pipeline run ID
print(f"Pipeline run ID: {pipeline_run.run_id}")
