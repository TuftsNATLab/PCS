#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 toy/themis1_scaled"
    exit 1
fi

# num_nodes
trace=$1

echo "Running trace $trace"



# PCS configs
for var in {jct,bal,pred,}
do
	output_file=new_data/PCS_"$var"_"$trace"_result.csv
	python3 testbed/run_experiment.py -trace trace_"$trace" -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_"$trace"_"$var".pkl -output_file "$output_file"
	echo "$var" done
done


# Other policies
for policy in {FIFO,SRSF,AFS,THEMIS,}
do
	output_file=new_data/"$policy"_"$trace"_result.csv
	python3 testbed/run_experiment.py -trace trace_"$trace" -scheduling_policy $policy -output_file $output_file
	echo "$policy" done
done

#Compare results
python3 new_data/parser.py -traces $trace