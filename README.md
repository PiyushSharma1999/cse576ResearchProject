# cse576ResearchProject

## How to run the project

1. Create a python virtual environment with the command `python3.11 -m venv venv`
2. Run the command to install requirements `pip install -r requirements.txt`
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

**Note:** 
1. Make sure to create the virtual environment with `python3.11` as some dependencies might fail with old versions.
2. Make sure to run executables such as pip, python, etc. from the `venv/bin`
