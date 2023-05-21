import docker

<<<<<<< HEAD
=======

>>>>>>> 976cf1406206f156ccb2b9255c65e390ab50179b
def test_docker_image(image_name):
    client = docker.from_env()

    try:
        container = client.containers.run(image_name, detach=True)
        logs = container.logs(stream=True)

        for log in logs:
            print(log.decode().strip())

<<<<<<< HEAD
        exit_code = container.wait()['StatusCode']
=======
        exit_code = container.wait()["StatusCode"]
>>>>>>> 976cf1406206f156ccb2b9255c65e390ab50179b
        container.remove()

        if exit_code == 0:
            print("Image test passed.")
        else:
            print(f"Image test failed with exit code: {exit_code}")
    except docker.errors.APIError as e:
        print(f"Error while running the container: {e}")

<<<<<<< HEAD
=======

>>>>>>> 976cf1406206f156ccb2b9255c65e390ab50179b
if __name__ == "__main__":
    image_name = "your-docker-image:tag"
    test_docker_image(image_name)
