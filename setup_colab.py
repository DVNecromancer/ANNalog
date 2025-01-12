import os
import subprocess

# Step 1: Define the private key for SSH
private_key = """-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAACFwAAAAdzc2gtcn
NhAAAAAwEAAQAAAgEA4OUj2w0VtPwQjjPqNUll2NApJ+eI0uANx+4vrya0Hqj0/bwGCp9L
...
-----END OPENSSH PRIVATE KEY-----"""

# Step 2: Configure SSH
print("Configuring SSH...")
os.makedirs('/root/.ssh', exist_ok=True)
with open('/root/.ssh/id_rsa', 'w') as key_file:
    key_file.write(private_key)

# Set permissions for the private key
subprocess.run(['chmod', '600', '/root/.ssh/id_rsa'])
with open('/root/.ssh/config', 'w') as config_file:
    config_file.write(
        "Host github.com\n\tStrictHostKeyChecking no\n\tIdentityFile ~/.ssh/id_rsa\n"
    )
subprocess.run(['chmod', '600', '/root/.ssh/config'])
subprocess.run(['ssh-keyscan', 'github.com'], stdout=open('/root/.ssh/known_hosts', 'a'))

# Test SSH connection
print("Testing SSH connection to GitHub...")
subprocess.run(['ssh', '-T', 'git@github.com'])

# Step 3: Clone the GitHub repository
print("Cloning the repository...")
subprocess.run(['git', 'clone', 'git@github.com:DVNecromancer/ANNalog.git'])

# Step 4: Install Conda via CondaColab
print("Installing Conda using CondaColab...")
subprocess.run(['pip', 'install', '-q', 'condacolab'])
import condacolab
condacolab.install()

# Step 5: Install gdown and download the installer script
print("Installing gdown and downloading the installer script...")
subprocess.run(['pip', 'install', 'gdown'])
subprocess.run(['gdown', '--id', '10bEXxjtPzgUXGDYEQvqB1gWFBkGiWWlT', '-O', 'final_seq_installer.sh'])

# Step 6: Run the installer script to set up the Conda environment
print("Running the installer script to set up the Conda environment...")
subprocess.run(['bash', 'final_seq_installer.sh', '-b', '-p', '/usr/local/envs/final_seq'])

# Step 7: Install the `annalog` package into the Conda environment
print("Installing the `annalog` package...")
subprocess.run([
    'conda', 'run', '-p', '/usr/local/envs/final_seq',
    'pip', 'install', '/content/ANNalog/annalog_package'
])

# Final step: Confirm the setup
print("Verifying the installation...")
subprocess.run([
    'conda', 'run', '-p', '/usr/local/envs/final_seq', 'python', '--version'
])
subprocess.run([
    'conda', 'run', '-p', '/usr/local/envs/final_seq', 'pip', 'list'
])
