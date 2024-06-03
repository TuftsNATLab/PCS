#!/bin/bash
traces=()


for trace in {gavel,}
do
	traces+=("$trace")
	
	for var in {jct,bal,pred,}
	do
		output_file=new_data/PCS_"$var"_"$trace"_result.csv
		python3 simulation/sim.py -workload $trace -load 1.2 -num_gpus 64 -num_apps 3000 -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_"$trace"_"$var".pkl  -output_file $output_file
		echo "PCS_$var" done
	done
		
	for policy in {FIFO,SRSF,AFS,THEMIS,}
	do
		output_file=new_data/"$policy"_"$trace"_result.csv
		python3 simulation/sim.py -workload $trace -load 1.2 -num_gpus 64 -num_apps 3000 -scheduling_policy $policy -output_file $output_file
		echo "$policy" done
	done
	
done

#Compare results
python3 new_data/parser.py -traces "${traces[@]}"