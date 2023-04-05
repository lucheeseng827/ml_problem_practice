import argparse
import subprocess
import sys


def list_instances(project, zone):
    command = f"gcloud compute instances list --project {project} --zone {zone}"
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE)
    print(result.stdout)


def create_instance(project, zone, instance_name):
    command = f"gcloud compute instances create {instance_name} --project {project} --zone {zone}"
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE)
    print(result.stdout)


parser = argparse.ArgumentParser(
    description="A script to interact with Google Cloud SDK"
)

parser.add_argument("-p", "--project", help="Google Cloud project ID", required=True)
parser.add_argument("-z", "--zone", help="Google Cloud zone", required=True)

subparsers = parser.add_subparsers(dest="command", required=True)

list_parser = subparsers.add_parser("list", help="List Compute Engine instances")

create_parser = subparsers.add_parser("create", help="Create a Compute Engine instance")
create_parser.add_argument("instance_name", help="Name of the new instance")

args = parser.parse_args()

if args.command == "list":
    list_instances(args.project, args.zone)
elif args.command == "create":
    create_instance(args.project, args.zone, args.instance_name)
else:
    sys.exit("Invalid command.")
