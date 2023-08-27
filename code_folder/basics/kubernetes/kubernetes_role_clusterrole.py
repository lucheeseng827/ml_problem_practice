from kubernetes import client, config


def create_role(namespace, role_name):
    config.load_kube_config()

    body = client.V1Role(
        api_version="rbac.authorization.k8s.io/v1",
        kind="Role",
        metadata=client.V1ObjectMeta(namespace=namespace, name=role_name),
        rules=[
            client.V1PolicyRule(
                api_groups=[""], resources=["pods"], verbs=["get", "list"]
            )
        ],
    )

    rbac_api = client.RbacAuthorizationV1Api()
    role = rbac_api.create_namespaced_role(namespace=namespace, body=body)
    print(f"Role created: {role.metadata.name}")


def create_cluster_role(cluster_role_name):
    config.load_kube_config()

    body = client.V1ClusterRole(
        api_version="rbac.authorization.k8s.io/v1",
        kind="ClusterRole",
        metadata=client.V1ObjectMeta(name=cluster_role_name),
        rules=[
            client.V1PolicyRule(
                api_groups=[""], resources=["nodes"], verbs=["get", "list"]
            )
        ],
    )

    rbac_api = client.RbacAuthorizationV1Api()
    cluster_role = rbac_api.create_cluster_role(body=body)
    print(f"ClusterRole created: {cluster_role.metadata.name}")


def validate_role(namespace, role_name):
    config.load_kube_config()

    rbac_api = client.RbacAuthorizationV1Api()
    try:
        rbac_api.read_namespaced_role(name=role_name, namespace=namespace)
        print(f"Role {role_name} exists in namespace {namespace}.")
    except client.rest.ApiException as e:
        print(f"Role {role_name} does not exist in namespace {namespace}.")


def validate_cluster_role(cluster_role_name):
    config.load_kube_config()

    rbac_api = client.RbacAuthorizationV1Api()
    try:
        rbac_api.read_cluster_role(name=cluster_role_name)
        print(f"ClusterRole {cluster_role_name} exists.")
    except client.rest.ApiException as e:
        print(f"ClusterRole {cluster_role_name} does not exist.")


if __name__ == "__main__":
    namespace = "default"
    role_name = "sample-role"
    cluster_role_name = "sample-clusterrole"

    create_role(namespace, role_name)
    validate_role(namespace, role_name)

    create_cluster_role(cluster_role_name)
    validate_cluster_role(cluster_role_name)
