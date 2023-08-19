import yaml
import subprocess

def git_commit_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    for update in data['updates']:
        name = update['name']

        try:
            command = 'git add'
            for file in update['files']:
                command += f" {file}"
            print(command)
            process = subprocess.Popen(command, shell=True)
            process.wait()


            command = f'git commit -m "{name}"'
            print(command)
            process = subprocess.Popen(command, shell=True)
            process.wait()

        except subprocess.CalledProcessError as e:
            print(f'Error in processing update {name}: {e}')
            break

# Call the function with your YAML file
git_commit_from_yaml('updates.yml')
