import os
import shutil
import git

def squash_commits(repository_path, branch_name):
    # Step 1: Clone the repository
    temp_dir = "temp_repo"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    repo = git.Repo.clone_from(repository_path, temp_dir)

    try:
        # Step 2: Check out the branch
        repo.git.checkout(branch_name)

        # Step 3: Perform an interactive rebase to squash commits
        squash_command = f"git rebase -i {branch_name}~{len(repo.head.commit.parents)}"
        repo.git.execute(squash_command)

        # Step 4: Amend the commit message (optional)
        # You can modify the commit message here if needed.
        # For automation, you can use `git.commit` method to programmatically amend the message.

        # Step 5: Perform a force push to update the remote repository with the squashed commit
        repo.git.push("origin", f"{branch_name}", force=True)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    repository_path = "https://github.com/your-username/your-repository.git"
    branch_name = "main"  # Replace with the branch name you want to squash

    squash_commits(repository_path, branch_name)
