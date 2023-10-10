import sys

import docker


def terminate_container(token):
    # Initialize a docker client
    client = docker.from_env()

    # List all containers
    containers = client.containers.list()

    # Find and stop the container with a matching name or ID
    for container in containers:
        if token in container.name or token in container.id:
            container.stop()
            print(
                f"Stopped container with ID: {container.id} and name: {container.name}"
            )
            return

    print(f"No container found for token: {token}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a token as an argument.")
        sys.exit(1)

    token = sys.argv[1]
    terminate_container(token)
