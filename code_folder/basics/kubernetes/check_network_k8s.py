import kubernetes
import logging



def check_container_network(kube_client, namespace, container_name):
    """
    Checks the network security of a container.

    Args:
      kube_client: A Kubernetes client object.
      namespace: The namespace of the container.
      container_name: The name of the container.

    Returns:
      A list of security issues, or an empty list if there are no issues.
    """

    # Get the container's network settings.
    container_network = kube_client.read_namespaced_pod_network_policy(
        namespace, container_name
    )

    # Check for common security issues.
    issues = []
    if container_network.host_network:
        issues.append("The container is using host networking, which is insecure.")
    if container_network.allow_external_connections:
        issues.append(
            "The container is allowing external connections, which is insecure."
        )
    if container_network.allow_connections_from_pods_in_other_namespaces:
        issues.append(
            "The container is allowing connections from pods in other namespaces, which is insecure."
        )

    return issues


def set_additional_network_latency(
    kube_client, namespace, pod_name, container_name, latency
):
    """
    Sets the additional network latency for a container.

    Args:
      kube_client: A Kubernetes client object.
      namespace: The namespace of the container.
      pod_name: The name of the pod.
      container_name: The name of the container.
      latency: The additional network latency in milliseconds.
    """

    # Get the pod's network settings.
    pod_network = kube_client.read_namespaced_pod_network_policy(namespace, pod_name)

    # Set the additional network latency.
    pod_network.additional_network_latency = latency

    # Update the pod's network settings.
    kube_client.update_namespaced_pod_network_policy(namespace, pod_name, pod_network)
