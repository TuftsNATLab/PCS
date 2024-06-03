python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py
sleep 10s
python3 run_experiment.py -scheduling_policy MCS -output_file MCS_3_1718,2757,inf_941,7,3_testbed.csv -from_trace 1 -MCS_config_file 3_1718,2757,inf_941,7,3.pkl

sleep 10s
python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py
sleep 10s
python3 run_experiment.py -scheduling_policy MCS -output_file MCS_2_4790,inf_23,2_testbed.csv -from_trace 1 -MCS_config_file 2_4790,inf_23,2.pkl

sleep 10s
python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py
sleep 10s
python3 run_experiment.py -scheduling_policy MCS -output_file MCS_2_3683,inf_461,39_testbed.csv -from_trace 1 -MCS_config_file 2_3683,inf_461,39.pkl

sleep 10s
python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py
sleep 10s
python3 run_experiment.py -scheduling_policy SRSF -output_file SRSF_testbed.csv -from_trace 1

sleep 10s
python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py
sleep 10s
python3 run_experiment.py -scheduling_policy AFS -output_file AFS_testbed.csv -from_trace 1

sleep 10s
python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py
sleep 10s
python3 run_experiment.py -scheduling_policy THEMIS -output_file THEMIS_testbed.csv -from_trace 1

sleep 10s
python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py
sleep 10s
python3 run_experiment.py -scheduling_policy FS -output_file FS_testbed.csv -from_trace 1

sleep 10s
python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py
sleep 10s
python3 run_experiment.py -scheduling_policy FIFO -output_file FIFO_testbed.csv -from_trace 1


# python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py

# python3 run_experiment.py -scheduling_policy MCS -output_file MCS_testbed_2_4200,inf_187,13.csv -from_trace 1 -MCS_config_file 2_4200,inf_187,13.pkl
# python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py
# sleep 20s

# python3 run_experiment.py -scheduling_policy MCS -output_file MCS_testbed_2_4503,inf_47,3.csv -from_trace 1 -MCS_config_file 2_4503,inf_47,3.pkl

# python3 /users/abdffsm/cloudlab_cluster_setup/setup_ray_cluster.py
# sleep 20s
# python3 run_experiment.py -scheduling_policy LAS -output_file LAS_testbed.csv -from_trace 1
#python3 run_experiment.py -scheduling_policy FIFO -output_file FIFO_testbed.csv -from_trace 1
