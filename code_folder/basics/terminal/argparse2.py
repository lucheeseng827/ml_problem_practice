import argparse
import subprocess
import sys


def list_ec2_instances(region):
    command = f"aws ec2 describe-instances --region {region}"
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE)
    print(result.stdout)


def create_ec2_instance(region, instance_type, ami_id):
    command = f"aws ec2 run-instances --region {region} --instance-type {instance_type} --image-id {ami_id}"
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE)
    print(result.stdout)


def list_virtual_machines(resource_group):
    command = f"az vm list -g {resource_group} -o table"
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE)
    print(result.stdout)


def create_virtual_machine(resource_group, vm_name, location):
    command = f"az vm create -g {resource_group} -n {vm_name} --location {location} --image UbuntuLTS --generate-ssh-keys"
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE)
    print(result.stdout)


parser = argparse.ArgumentParser(
    description="A script to interact with AWS CLI and Azure CLI"
)

parser.add_argument(
    "-p", "--provider", help="Cloud provider (aws or azure)", required=True
)

subparsers = parser.add_subparsers(dest="command", required=True)

list_parser = subparsers.add_parser("list", help="List instances")
list_parser.add_argument(
    "-r",
    "--region",
    help="Region (for AWS) or resource group (for Azure)",
    required=True,
)

create_parser = subparsers.add_parser("create", help="Create an instance")
create_parser.add_argument(
    "-r",
    "--region",
    help="Region (for AWS) or resource group (for Azure)",
    required=True,
)
create_parser.add_argument(
    "-n", "--name", help="Name of the new instance (for Azure only)"
)
create_parser.add_argument(
    "-i", "--image", help="Image ID (for AWS) or location (for Azure)"
)

args = parser.parse_args()

if args.provider == "aws":
    if args.command == "list":
        list_ec2_instances(args.region)
    elif args.command == "create":
        if not args.image:
            sys.exit("Image ID is required for AWS.")
        create_ec2_instance(args.region, args.image)
    else:
        sys.exit("Invalid command.")
elif args.provider == "azure":
    if args.command == "list":
        list_virtual_machines(args.region)
    elif args.command == "create":
        if not args.name or not args.image:
            sys.exit("Name and location are required for Azure.")
        create_virtual_machine(args.region, args.name, args.image)
    else:
        sys.exit("Invalid command.")

else:
    sys.exit("Invalid provider.")
