from kubernetes import client, config


def check_kubernetes_status():
    try:
        config.load_kube_config()  # Loads the Kubernetes configuration from the default location
        api_instance = client.CoreV1Api()

        # Check if the Kubernetes API server is reachable
        api_response = api_instance.get_api_versions()
        if api_response:
            print("Kubernetes API server is online and accessible.")
        else:
            print(
                "Kubernetes API server is online, but the response indicates an error."
            )

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Kubernetes API server is unreachable.")


# Usage example
check_kubernetes_status()
