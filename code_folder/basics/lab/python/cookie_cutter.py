import os


## make sure you are in the correct directory
# os.chdir("code_folder/basics/lab/python")
def create_file(path, content=""):
    with open(path, "w") as file:
        file.write(content)


def create_directory_structure(project_name):
    # Create main project directory
    os.makedirs(project_name, exist_ok=True)

    # Create src and tests directories inside project directory
    os.makedirs(os.path.join(project_name, "src"), exist_ok=True)
    os.makedirs(os.path.join(project_name, "tests"), exist_ok=True)

    # Create a sample config.cfg file
    config_content = """
[DEFAULT]
setting1 = value1
setting2 = value2
    """
    create_file(os.path.join(project_name, "config.cfg"), config_content)

    # Create a sample settings.yaml file
    yaml_content = """
settings:
    - key: value
    """
    create_file(os.path.join(project_name, "settings.yaml"), yaml_content)

    print(f"Project {project_name} has been set up!")


if __name__ == "__main__":
    project_name = input("Enter the name of the project: ")
    create_directory_structure(project_name)


"""
project folder structure:

/my_project
    /src
    /tests
    config.cfg
    settings.yaml
"""
