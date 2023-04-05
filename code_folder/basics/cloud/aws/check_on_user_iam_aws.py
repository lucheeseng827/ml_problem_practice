import boto3

iam = boto3.client("iam")

# List of user names
user_list = ["user1", "user2", "user3"]

# Iterate over each user in the list
for user in user_list:
    print(f"Roles for user {user}:")
    response = iam.list_roles_for_instance_profile(InstanceProfileName=user)
    roles = response["Roles"]
    if len(roles) == 0:
        print("No roles found")
    else:
        for role in roles:
            print(role["RoleName"])
