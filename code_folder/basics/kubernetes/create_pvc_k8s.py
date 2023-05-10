from kubernetes import client, config


def create_pvc():
    config.load_kube_config()

    api = client.CoreV1Api()

    body = client.V1PersistentVolumeClaim(
        api_version="v1",
        kind="PersistentVolumeClaim",
        metadata=client.V1ObjectMeta(name="my-pvc"),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            resources=client.V1ResourceRequirements(requests={"storage": "1Gi"}),
        ),
    )

    api.create_namespaced_persistent_volume_claim(namespace="default", body=body)


if __name__ == "__main__":
    create_pvc()
