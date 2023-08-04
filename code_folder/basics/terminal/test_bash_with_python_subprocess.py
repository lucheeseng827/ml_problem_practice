import subprocess


def test_run_specific_command():
    argument = "example_argument"
    command = ["bash", "your_script.sh", argument]

    # Run the command and capture the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Assert the desired behavior
    assert process.returncode == 0  # Check if the command executed successfully
    assert (
        b"Expected output" in stdout
    )  # Check if the expected output is present in stdout
    assert stderr == b""  # Check if there is no error message

    # Add more assertions as needed


# Run the test
test_run_specific_command()
