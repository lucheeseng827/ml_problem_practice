#!/bin/bash

# Define your resources here (add or remove resources as needed)
resources=(
  "aws_vpc"
  "aws_subnet"
  "aws_security_group"
  "aws_instance"
)

# Create project folder and subdirectories
project_name="terraform-project"
mkdir -p "${project_name}/modules" "${project_name}/environments"

# Create a README file for the project
readme_file="${project_name}/README.md"
touch $readme_file

# Add a header and description to the README file
echo "# Terraform Project" >> $readme_file
echo "This Terraform project contains the following resources:" >> $readme_file

# Create a Terraform module for each resource and update the README file
for resource in "${resources[@]}"; do
  # Create a folder for the module
  mkdir "${project_name}/modules/${resource}"

  # Create main.tf, variables.tf, and outputs.tf files for the module
  touch "${project_name}/modules/${resource}/main.tf"
  touch "${project_name}/modules/${resource}/variables.tf"
  touch "${project_name}/modules/${resource}/outputs.tf"

  # Add the resource to the README file
  echo "- ${resource}" >> $readme_file
done

# Create a main.tf file for the environments folder
touch "${project_name}/environments/main.tf"

echo "The project folder structure has been created successfully."
