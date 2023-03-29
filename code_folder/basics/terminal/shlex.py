import subprocess
import shlex

gcloud_args = ["gcloud", "some-command", "--some-flag", "some-value"]
quoted_args = [shlex.quote(arg) for arg in gcloud_args]
command = ' '.join(quoted_args)
# command = shlex.join(gcloud_args) # Python 3.8+

subprocess.run(command, shell=True, check=True)
