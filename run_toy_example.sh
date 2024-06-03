#!/bin/bash
traces=()


for trace in {themis1,}
do
	traces+=("$trace")
	
	for var in {jct,bal,pred,}
	do
		output_file=new_data/PCS_"$var"_"$trace"_result.csv
		python3 simulation/sim.py -workload $trace -num_gpus 64 -num_apps 500 -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_"$trace"_"$var".pkl -output_file $output_file
		echo "PCS_$var" done
	done
		
	for policy in {FIFO,SRSF,AFS,THEMIS,}
	do
		output_file=new_data/"$policy"_"$trace"_result.csv
		python3 simulation/sim.py -workload $trace -num_gpus 64 -num_apps 500 -scheduling_policy $policy -output_file $output_file
		echo "$policy" done
	done
	
done

#Compare results
python3 new_data/parser.py -traces "${traces[@]}"