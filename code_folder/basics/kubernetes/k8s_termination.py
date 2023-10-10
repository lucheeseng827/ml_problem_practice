import sys

from kubernetes import client, config


def terminate_pod(token):
    # Load the kube config from the default location (e.g., ~/.kube/config)
    config.load_kube_config()

    # Initialize the API client
    v1 = client.CoreV1Api()

    # List all the pods
    ret = v1.list_pod_for_all_namespaces(watch=False)

    for pod in ret.items:
        if token in pod.metadata.name or token in pod.metadata.uid:
            print(
                f"Found pod with name: {pod.metadata.name} in namespace: {pod.metadata.namespace}. Terminating..."
            )

            # Delete the pod
            v1.delete_namespaced_pod(
                name=pod.metadata.name, namespace=pod.metadata.namespace
            )
            return

    print(f"No pod found for token: {token}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a token as an argument.")
        sys.exit(1)

    token = sys.argv[1]
    terminate_pod(token)
