#!/bin/bash
traces=()


# Set num apps between 100-500 for testing. -1 means the entire trace
num_apps=-1

for trace in {0e4a51,ee9e8c,}
# for trace in {0e4a51,6214e9,6c71a0,b436b2,ee9e8c,}
do
	traces+=("$trace")
	
	for var in {jct,bal,pred,}
	do
		output_file=new_data/PCS_"$var"_"$trace"_result.csv
		python3 simulation/sim.py -trace trace_"$trace" -num_gpus 64 -num_apps $num_apps -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_"$trace"_"$var".pkl -output_file $output_file
		echo "PCS_$var" done
	done
		
	for policy in {FIFO,SRSF,AFS,THEMIS,}
	do
		output_file=new_data/"$policy"_"$trace"_result.csv
		python3 simulation/sim.py -trace trace_"$trace" -num_gpus 64 -num_apps $num_apps -scheduling_policy $policy -output_file $output_file
		echo "$policy" done
	done
	
done

#Compare results
python3 new_data/parser.py -traces "${traces[@]}"