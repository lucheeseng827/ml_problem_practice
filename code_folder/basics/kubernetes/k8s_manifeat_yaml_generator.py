import os

# Load configuration from config.py
from kubernetes import client as clt
from kubernetes import config

# define DEPLOYMENT_IMAGE, DEPLOYMENT_NAME, NAMESPACE, REPLICAS
DEPLOYMENT_IMAGE = "nginx:1.7.9"
DEPLOYMENT_NAME = "nginx-deployment"
NAMESPACE = "default"
REPLICAS = 3

# Load kube config
config.load_kube_config(os.path.join(os.environ["HOME"], ".kube/config"))

# Create an instance of the API class
api_instance = clt.AppsV1Api()

# Define the pod spec
pod_spec = clt.V1PodSpec(
    containers=[clt.V1Container(name=DEPLOYMENT_NAME, image=DEPLOYMENT_IMAGE)]
)

# Define the pod template spec
pod_template_spec = clt.V1PodTemplateSpec(
    metadata=clt.V1ObjectMeta(labels={"app": DEPLOYMENT_NAME}), spec=pod_spec
)

# Define the deployment spec
deploy_spec = clt.V1DeploymentSpec(
    replicas=REPLICAS,
    template=pod_template_spec,
    selector={"matchLabels": {"app": DEPLOYMENT_NAME}},
)

# Define the deployment
deployment = clt.V1Deployment(
    api_version="apps/v1",
    kind="Deployment",
    metadata=clt.V1ObjectMeta(name=DEPLOYMENT_NAME),
    spec=deploy_spec,
)

# Create deployment
api_response = api_instance.create_namespaced_deployment(
    body=deployment, namespace=NAMESPACE
)

print(f"Deployment {DEPLOYMENT_NAME} created.")
