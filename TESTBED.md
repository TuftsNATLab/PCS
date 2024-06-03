# Follow these instructions for setting up the GPU testbed on cloudlab or locally


# Local Setup
* Run `git clone https://github.com/abdullahfsm/PCS.git`, navigate to the repo `PCS/`, run `git checkout osdi2024-artifact` and finally navigate to `utils`
* Run `python3 local_cluster_utils.py install` This script takes a considerable time (~20 min) to install the required dependencies and upon successfully completing will reboot your machine
* Once your machine is rebooted:
* * Naviagate to `PCS/utils`
* * Run `python3 local_cluster_utils.py launch` This will set up a ray cluster!
* * To verify that the cluster has successfully launched, run `ray status` which will show the resources and nodes available to the ray cluster. We are now ready to run our workloads!


# CloudLab Setup
## Starting a cloudlab experiment
Assuming that you have a cloudlab profile:
* Navigate to [Create Experiment Profile](https://www.cloudlab.us/manage_profile.php)
* Upload the provided cloudlab profile (utils/cloudlab_profile.py)
* Once the experiment profile has been set up, navigate to [Start Experiment](https://www.cloudlab.us/instantiate.php) and choose the profile just created
* For reproducing results, the parameters should be `n: 16` and `ntype: c240g5`, otherwise, the default should be sufficient
* Starting the experiment may take some time
* Once the experiment is up and running you should be able to ssh into the head node (n0)


## Setting up cloudlab cluster
Assuming you can ssh into the head node of a cloudlab cluster:
* Run `git clone https://github.com/abdullahfsm/PCS.git` at root directory followed by `cd ~/PCS`, `git checkout osdi2024-artifact` and finally `cd utils`
* Run `python3 cluster_utils.py install` This script takes a considerable time (~20 min) to install the required dependencies and upon successfully completing will reboot all the nodes in the cluster (you will have to login to the cloudlab head node again)
* Once all the cluster machines have been rebooted:
* * Login to the cloudlab head node again, navigate to `cd ~/PCS/utils`
* * Run `python3 cluster_utils.py launch` This will set up a ray cluster!
* * To verify that the cluster has successfully launched, run `ray status` which will show the resources and nodes available to the ray cluster. We are now ready to run our workloads!

# Experiments
## Running a Toy testbed experiment
* Run `bash run_testbed.sh toy`


## Reproducing Figure 4-6 Data
* Assuming you have a ray cluster with 16 c240g5 type nodes up and running: Run `bash run_testbed.sh themis1_scaled`
* This can take a considerable amount of time to run ~5-7 hours for each policy (there are a total of 7 policies by default)