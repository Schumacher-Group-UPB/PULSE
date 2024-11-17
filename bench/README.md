# Benchmarking

## Setting up a Benchmark Series
### Configuration
A series of benchmarks is defined by a json file like
```
{
  "commits":[
    {"label":"bench","repo":"https://github.com/davidbauch/PHOENIX/","type":"branch"},
    {"label":"4fa2c3b","repo":"https://github.com/davidbauch/PHOENIX/","type":"commit"},
  ],
  "envs":
  [
    {"label":"gcc-13.2.0","modules":["system/CUDA/12.6.0","tools/likwid/5.3.0-GCC-13.2.0","toolchain/foss/2023b","lang/Python/3.12.3-GCCcore-13.2.0"],"compiler":"g++","likwid":"likwid-perfctr","machinestate":"~/.local/bin/machinestate","nvidia-smi":"nvidia-smi","perf":"perf"}
  ],
  "makes":
  [
    {"label":"full","cmd":"make SFML=FALSE FP32=TRUE TETM=FALSE CPU=TRUE COMPILER=_compiler_ NO_CALCULATE_K=FALSE NO_INTERMEDIATE_SUM_K=FALSE NO_FINAL_SUM_K=FALSE NO_HALO_SYNC=FALSE AVX2=TRUE BENCH=TRUE LIKWID=TRUE"},                                   
    {"label":"halo_sync","cmd":"make SFML=FALSE FP32=TRUE TETM=FALSE CPU=TRUE COMPILER=_compiler_ NO_CALCULATE_K=TRUE NO_INTERMEDIATE_SUM_K=TRUE NO_FINAL_SUM_K=TRUE NO_HALO_SYNC=FALSE AVX2=TRUE BENCH=TRUE LIKWID=TRUE"}                                  
  ],
  "threads": [64],
  "grids": [1024,2048],
  "subgrids": [[8,8],[16,16]],
  "likwid_metrics": ["ENERGY","FLOPS_SP"]
}
```

The set of benchmarks cases to be run is the product of all lists of fields. 

That means that in the above config in total 32 variants are run:

* 2 versions of the code (branch bench and the commit 4fa2c3b)
* 1 choice of the environemnt in terms of compilers
* 2 variants of the code, one `full` and one that only contains the halo synchronization
* 1 nums of threads
* 2 choices of grid sizes (1024 and 2048)
* 2 choices of subgrids (8x8 and 16x16)
* 2 choices of performance metrics to be recorded (`ENERGY` and `FLOPS_SP`)

To run these versions in total 4 variants of the solver have to be compiled:
* 2 versions of the code (branch bench and the commit 4fa2c3b)
* 1 choice of the environemnt in terms of compilers
* 2 variants of the code, one `full` and one that only contains the halo synchronization

### Collecting Additional Information
During a benchmark run additional information can be collected with tools. This is configured via the `envs` field. To deactive a tool, remove the mention from the `envs` field. The entry in the `envs`-fields are the commands to run the tools.

#### Machinestate
Machinestate (https://github.com/RRZE-HPC/MachineState) is a nice tool to automatically document the current system settings and the execution environment in terms of software and hardware for a benchmark run. The result is a json file `machinestate.json` in the subdirectory of the benchmark run.

#### LIKWID
LIKWID (https://github.com/RRZE-HPC/likwid) is a nice tool to access hardware-performance counter during a benchmark run to a measure for example the flaoting point throughput or cache utilization. The result is a json file `likwid.json` in the subdirectory of the benchmark run. Due to limitations of the number of registers for the hardwrae performance counter not all metrics can be measured at the same time. Thus, the full set is measured with multiple benchmark runs instead. Important metrics are directly included in the main result file `run.json` in the subdirectory of the benchmark run. (`cpu_energy_J_likwid` only has a valid value if in this run the `ENERGY` metric is measured with LIKWID.)

#### Linux Perf
In additiog to LIKWID (or if LIKWID is not available) the energy consumption of the CPU sockets can be measured with the Linux perf interface via RAPL energy counter. The result file `perf.out` contains the energy consumption of the CPU sockets during the calculation. The energy usage is included in the main result file `run.json` as `cpu_energy_J_perf` in the subdirectory of the benchmark run.

#### nvidia-smi
nvidia-smi can be used to measure, for exmaple, the power usage of NVIDIA GPUs. The result file `nvidia.csv` contains the energy consumption of the GPU sockets during the calculation. The energy usage is included in the main result file `run.json` as `gpu_energy` in the subdirectory of the benchmark run.

## Running a Benchmark
The benchmark script `bench.py` will perform the steps (compiling, running and summarizing) automatically. It will create the directories:
* `runs/[name of the json config file]/0`, `runs/[name of the json config file]/1`,... for the benchmark runs defined by the json config
* `bins/build_[commit label]_[env label]_[make label]`,... for the binaries used for the benchmark runs

It should be run as `python bench.py -c [json config file]`.

For each benchmark run, the main result file `run.json` holds all relevant results
```
{
    "bindir": "/scratch/pc2-mitarbeiter/rschade/tickets/pulse/PHOENIX/bench/build_bench_gcc-13.2.0_full", 
    "wall": 10.0, 
    "ms/ps": 0.261, 
    "ps/s": 3830.0, 
    "mus/it": 35.8, 
    "gpu_energy_J": 0, 
    "cpu_energy_J_perf": 2611.95, 
    "cpu_energy_J_likwid": 0, 
    "likwid_measurement": 296446.8005, 
    "commits": {"label": "bench", "repo": "https://github.com/davidbauch/PHOENIX/", "type": "branch"}, 
    "envs": {
        "label": "gcc-13.2.0", 
        "modules": ["system/CUDA/12.6.0", "tools/likwid/5.3.0-GCC-13.2.0", "toolchain/foss/2023b", "lang/Python/3.12.3-GCCcore-13.2.0"], 
        "compiler": "g++", 
        "likwid": "likwid-perfctr", 
        "machinestate": "~/.local/bin/machinestate", 
        "nvidia-smi": "nvidia-smi", 
        "perf": "perf"
    }, 
    "makes": {
        "label": "full", 
        "cmd": "make SFML=FALSE FP32=TRUE TETM=FALSE CPU=TRUE COMPILER=_compiler_ NO_CALCULATE_K=FALSE NO_INTERMEDIATE_SUM_K=FALSE NO_FINAL_SUM_K=FALSE NO_HALO_SYNC=FALSE AVX2=TRUE BENCH=TRUE LIKWID=TRUE"
    }, 
    "threads": 64, 
    "grids": 192, 
    "subgrids": [8, 8], 
    "likwid_metrics": "FLOPS_SP"
}
```

## Creating CSV-Files
A csv file can be produced from the benchmark runs for example with
```
python3 bench.py -c noctua2_7763.json --dryrun --fields=grids,subgrids,mus/it,likwid_metrics,likwid_measurement,threads=64 --output grids_7763.csv
```
The meaning of the command line arguments is:

* `--dryrun` or `-d`: doesn't build and run the benchmarks but just uses the data availbale in the directory structure
* `--output` or `-o`: path of the csv output file
* `--fields` or `-f`: specifies the fields to be included in the csv. Filters can be defined with `fieldname=value`

## Plotting
To recreate the plots shown in the paper, first decompress the data files in the `runs` directory and then run the Pyhton script `plots.py`.
