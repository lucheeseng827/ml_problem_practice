from kubernetes import client, config


def list_crds_in_namespace(namespace="default"):
    config.load_kube_config()

    api_instance = client.CustomObjectsApi()

    group = "apiextensions.k8s.io"
    version = "v1"
    plural = "customresourcedefinitions"

    try:
        api_response = api_instance.list_namespaced_custom_object(
            group, version, namespace, plural
        )
        return api_response["items"]
    except client.ApiException as e:
        print(
            f"Exception when calling CustomObjectsApi->list_namespaced_custom_object: {e}"
        )
        return []


def validate_crd(crd):
    # Placeholder function to validate CRD. Here, you can add the logic to check the specifics
    # of the CRD based on your requirements.
    # Currently, it just checks if 'spec' and 'names' are present.
    return "spec" in crd and "names" in crd


def main():
    namespace = "default"  # Adjust namespace as needed
    crds = list_crds_in_namespace(namespace)

    for crd in crds:
        crd_name = crd["metadata"]["name"]
        if validate_crd(crd):
            print(f"CRD {crd_name} in namespace {namespace} is valid.")
        else:
            print(f"CRD {crd_name} in namespace {namespace} is INVALID.")


if __name__ == "__main__":
    main()
