import kfp
from kfp import dsl


@dsl.pipeline(
    name='My Dataflow Pipeline',
    description='A pipeline that processes data using Dataflow.'
)
def my_dataflow_pipeline(
    input_bucket: str,
    output_bucket: str,
    keyword: str,
    project_id: str,
    region: str,
    num_workers: int,
    machine_type: str,
):
    """The main function that defines the Kubeflow Pipeline."""

    setup_bucket_op = dsl.ContainerOp(
        name='setup-bucket',
        image='gcr.io/google.com/cloudsdktool/cloud-sdk:alpine',
        command=['bash', '-c'],
        arguments=['gsutil mb -p $0 -l $1 $2', project_id,
                   region, input_bucket, output_bucket],
        file_outputs={
            'input_bucket': '/output/input_bucket.txt',
            'output_bucket': '/output/output_bucket.txt',
        },
    )

    dataflow_op = dsl.ContainerOp(
        name='run-dataflow-job',
        image='gcr.io/dataflow-templates-base/python3-template-launcher:latest',
        arguments=[
            '--runner=DataflowRunner',
            '--project=$0',
            '--region=$1',
            '--temp_location=$2/temp',
            '--staging_location=$2/staging',
            '--template_location=$2/template',
            '--num_workers=$3',
            '--worker_machine_type=$4',
            '--input=$5',
            '--output=$6',
            '--keyword=$7',
        ],
        file_outputs={
            'output': '/output.txt',
        },
    )

    setup_bucket_op.container.set_image_pull_policy('Always')
    setup_bucket_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    setup_bucket_op.execution_options.caching_strategy.enable_cache = True

    dataflow_op.container.set_image_pull_policy('Always')
    dataflow_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    dataflow_op.execution_options.caching_strategy.enable_cache = True

    setup_bucket_op.set_outputs(['input_bucket', 'output_bucket'])

    dataflow_op.apply(gcp.use_gcp_secret('user-gcp-sa'))

    dataflow_op.after(setup_bucket_op)

    dataflow_op.set_memory_request('4G')
    dataflow_op.set_memory_limit('4G')
    dataflow_op.set_cpu_request('1')
    dataflow_op.set_cpu_limit('1')

    dataflow_op.container.add_env_variable(
        k8s_client.V1EnvVar(
            name='GOOGLE_APPLICATION_CREDENTIALS',
            value='/secret/gcp-credentials/user-gcp-sa.json',
        )
    )

    dataflow_op.container.add_volume_mount(
        k8s_client.V1VolumeMount(
            name='gcp-credentials',
            mount_path='/secret/gcp-credentials',
        )
    )
