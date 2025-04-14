# cse576ResearchProject

Branch naming convention
asurite/task-name
ex: nsano1/initial-draft

## How to run the project

1. Create a python virtual environment with the command `python3.11 -m venv venv`
2. Run the command to install requirements `pip install -r requirements.txt` For windows: `./venv/bin/pip install -r requirements.txt`
3. Run `app.py`

## SOL Guide

1. Login to SOL (make sure you are connected to VPN domain: sslvon.asu.edu through cisco secure client)
2. Click on Interactive Apps
3. Click on VSCode Tunnel

Account: class_cse576spring2025

Partition: general

QOS: class

CPU Core Allocation: 32

Memory Allocation: 84

GPU Resources: gpu:a100:1

VSCode Tunnel Wall Time: As per requirement (check time format. For hours format is HH:MM:SS->10:00:00)

VSCode Instance: Primary

4. Click Launch

5. Wait for instance to start
6. Click Open in VSCode Desktop or Web
7. Once VSCode opens login via GiHub if prompted 
8. Click >< symbol (bottom left)
9. Click Connect to Tunnel
10. Run `module load cuda-12.6.1-gcc-12.1.0`
11. Run `nvcc --version` to check if cuda is loaded
12. Run `module load gcc-13.2.0-gcc-12.1.0`

## Hugging face CL guide
1. Click profile picture on hugging face
2. Go to “access tokens”
3. Create new token
4. Name it anything
5. Check “Read access to contents of all repos under your personal namespace”
6. In repositories permissions add the wanted model names in “search for repos”
7. Check “Read access to contents of selected repos”
8. Create token
9. Save the token somewhere safe
10. For models like Mistal AI, go back to the original models page and accept a term in order to fully be granted access. For other models, also go back to the original model page, submit an access request, and it must be reviewed before access is fully granted. 
11. Make sure you are in the virtual environment “source ./venv/bin/activate”
12. Run“huggingface-cli login” 
13. Copy and paste the token when asked.
14. Now, in the project terminal do the following installations
15. pip install protobuf
16. pip install sentencepiece
17. Now you should be able to add the hugging face model and utilize it.


**Note:** 
1. Make sure to create the virtual environment with `python3.11` as some dependencies might fail with old versions.
2. Make sure to run executables such as pip, python, etc. from the `venv/bin` For windows: `./venv/bin/python main/app.py`


**Another note:**
Go to config.py and change the model name if wanting to use a different model - read through model.py to see requirements


