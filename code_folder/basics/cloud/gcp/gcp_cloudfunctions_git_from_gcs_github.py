import os
import subprocess

from google.cloud import storage


def github_commit(event, context):
    # Get the bucket and file information from the event payload
    bucket_name = event["bucket"]
    file_name = event["name"]

    # Create a client to interact with the Cloud Storage API
    storage_client = storage.Client()

    # Download the GitHub token file from the specified bucket and file path
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    local_file_path = (
        "/tmp/github_token.txt"  # Set the local file path to store the token file
    )
    blob.download_to_filename(local_file_path)

    # Read the GitHub token from the downloaded file
    with open(local_file_path, "r") as file:
        github_token = file.read().strip()

    # Set the repository information
    repo_owner = "your-github-username"
    repo_name = "your-repository-name"

    # Set the git commands
    git_commands = [
        ["git", "config", "--global", "user.email", "your-email@example.com"],
        ["git", "config", "--global", "user.name", "Your Name"],
        ["git", "clone", f"https://github.com/{repo_owner}/{repo_name}.git"],
        ["git", "add", "."],
        ["git", "commit", "-m", "Commit from Cloud Function"],
        ["git", "push"],
    ]

    # Change the current working directory to the cloned repository directory
    os.chdir(f"/tmp/{repo_name}")

    # Execute the git commands
    for command in git_commands:
        subprocess.run(command, check=True)

    print("Git operations completed successfully.")


# Sample event payload:
event_payload = {"bucket": "your-bucket-name", "name": "github_token.txt"}

# Call the function for testing
github_commit(event_payload, None)
