from kubernetes import client, config


def create_deployment_object():
    # Configure Pod template container
    container = client.V1Container(
        name="my-container",
        image="my-image:latest",
        security_context=client.V1SecurityContext(
            run_as_user=1000,
            allow_privilege_escalation=False,
        ),
    )

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "my-app"}),
        spec=client.V1PodSpec(containers=[container]),
    )

    # Specify the Deployment spec
    spec = client.V1DeploymentSpec(
        replicas=3,
        selector=client.V1LabelSelector(match_labels={"app": "my-app"}),
        template=template,
    )

    # Create and configure the Deployment
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name="my-deployment"),
        spec=spec,
    )

    return deployment


def create_deployment(api_instance, deployment):
    # Actually create the Deployment
    api_response = api_instance.create_namespaced_deployment(
        body=deployment, namespace="default"
    )
    print("Deployment created. Status='%s'" % str(api_response.status))


def main():
    # Load kubeconfig
    config.load_kube_config()

    # Create a deployment object
    deployment = create_deployment_object()

    # API instance for deployments
    api_instance = client.AppsV1Api()

    # Create Deployment
    create_deployment(api_instance, deployment)


if __name__ == "__main__":
    main()
