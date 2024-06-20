# Reproducing Results (OSDI'24)


## Testbed setup
* For testbed setup and experiments please refer to `TESTBED.md`

## System Prerequisites 

* This has been tested on Ubuntu 22.04 with Python 3.10.12. 
* It is assumed that Python 3.10.12 can be invoked with `python3`.

## System Setup

* Run `bash setup.sh` and follow all prompts. 

## Regenerating Figures

* Run `jupyter notebook regenerate_plots.ipynb` and execute the cells to (re)generate plots from our paper


## Running a Single simulation

* Run `python3 simulation/sim.py -workload themis1 -num_gpus 64 -num_apps 500 -scheduling_policy FIFO -output_file FIFO_toy_result.csv` to run a simulation with a FIFO scheduler, 500 applications sampled from the themis workload on a 64 GPU cluster
* Run `python3 simulation/utils/result_summary.py -fnames FIFO_toy_result.csv` to get the summary statistics

## Running a Toy simulation

* Run `bash run_toy_example.sh` to run and compare FIFO, SRSF, PCS-pred, PCS-bal, PCS-jct with the same workload/configurations as in the previous example


## Reproducing Figure 7 Data
* Run `bash run_workload2.sh`
* Once the script finishes, you can use `figure7_plot.ipynb` to analyze and plot figure 7a,7b

## Reproducing Figure 8 Data
* Run `bash run_workload3.sh`
* Once the script finishes, you can use `figure8_plot.ipynb` to analyze and plot figure 8a,8b

## Reproducing Figure 9 Data
* Run `bash profile_time_per_sim.sh`
* Once the script finishes, you can use `figure9a_plot.ipynb` to analyze and plot figure 9a

## Reproducing Figure 10 Data
* Run `bash profile_sensitivity_error_in_size.sh`
* Once the script finishes, you can use `figure10_plot.ipynb` to analyze and plot figure 10a,10b


## Running Simulation 

* Run `python3 reproduce.py -h` to see which traces and workloads are acceptable for correct usage. 
* It will also show you additional simulation configuration parameters.

## Paper
The PDF of the paper is also available


# Navigating the Interface

* Upon running `python3 reproduce.py <workload>`, you will be shown an interactive figure showing the pareto front. 
* Click a point on the pareto front and click `Create Config`. This will generate a configuration file for a simulation which will then run. The simulation results are displayed at the end.
