import git


def commit_and_push(repo_path, commit_message):
    # Access the Git repository
    repo = git.Repo(repo_path)

    # Check for uncommitted changes
    if repo.is_dirty():
        # Stage all changes
        repo.git.add(A=True)

        # Commit the changes
        repo.index.commit(commit_message)

        # Push the changes to the remote repository
        origin = repo.remote("origin")
        origin.push()

    else:
        print("No changes to commit")


if __name__ == "__main__":
    repo_path = "/path/to/your/repo"
    commit_message = "Commit and push via Python SDK"
    commit_and_push(repo_path, commit_message)


"""
pip install gitpython
git config --global user.name "Your Name"
git config --global user.email "youremail@example.com"
"""
