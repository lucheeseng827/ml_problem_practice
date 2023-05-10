import yaml
from kubernetes import client, config


def list_pod_labels(namespace):
    config.load_kube_config()

    api = client.CoreV1Api()

    pod_list = api.list_namespaced_pod(namespace)
    labels = {pod.metadata.name: pod.metadata.labels for pod in pod_list.items}

    with open("pod_labels.yaml", "w") as outfile:
        yaml.dump(labels, outfile, default_flow_style=False)


if __name__ == "__main__":
    list_pod_labels("default")
