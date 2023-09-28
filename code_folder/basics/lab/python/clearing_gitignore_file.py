import fnmatch
import os


def find_files_to_delete(directory, gitignore_patterns):
    files_to_delete = []

    for root, _, filenames in os.walk(directory):
        for pattern in gitignore_patterns:
            for filename in fnmatch.filter(filenames, pattern):
                files_to_delete.append(os.path.join(root, filename))

    return files_to_delete


def main():
    with open(".gitignore", "r") as f:
        gitignore_patterns = f.readlines()

    gitignore_patterns = [
        pattern.strip()
        for pattern in gitignore_patterns
        if pattern and not pattern.startswith("#")
    ]

    files_to_delete = find_files_to_delete(os.getcwd(), gitignore_patterns)

    if not files_to_delete:
        print("No files match the patterns in .gitignore.")
        return

    print("Found files to delete:")
    for file_path in files_to_delete:
        print(f"- {file_path}")

    confirmation = input("Do you want to delete these files? (y/N): ").lower()
    if confirmation == "y":
        for file_path in files_to_delete:
            os.remove(file_path)
        print("Files deleted.")
    else:
        print("Files not deleted.")


if __name__ == "__main__":
    main()
