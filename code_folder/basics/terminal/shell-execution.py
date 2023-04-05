# import subprocess

# command = ['echo', 'Hello, World!']
# result = subprocess.run(command, capture_output=True)
# result = subprocess.run(command, capture_output=True, text=True)
# # Access captured output
# print(result.stdout)

import subprocess
import sys

if sys.platform == "win32":
    command = ["cmd.exe", "/c", "gcloud", "projects", "list", "--format", "json"]
else:
    command = ["echo", "Hello, World!"]

result = subprocess.run(command, capture_output=True, text=True)
print(result.stdout)
