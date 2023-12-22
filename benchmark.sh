#!/bin/bash
#export CUDA_VISIBLE_DEVICES=1

output_path="b_benchmarks/benchmark_3070ti_fixed/"
N_list=(100 200 300 400 500 600 700 800 1000 1200 1400 1600)
makef=true

system_parameters=(
    "--pump 40 add 30 0 0 plus 1 none gauss" # Center Pump
    "--pump -1 add+adaptive 2.5 0 15 plus 5 none gauss+outerExponent" # Potential Well
    "--pump -1 add+adaptive 2.5 0 7.5 plus 5 none gauss+outerExponent" # Potential Well
    "--pump -0.975 add+adaptive 2.5 0 0 plus 5 none gauss+outerExponent" # Potential Well
    "--pump -0.95 add+adaptive 2.5 0 -7.5 plus 5 none gauss+outerExponent" # Potential Well
    "--pump -0.93 add+adaptive 2.5 0 -15 plus 5 none gauss+outerExponent" # Potential Well
    "--pump 10 add 2.5 0 19 plus 1 none gauss" # Narrow
    "--outEvery 200"
    #"--pulse 10 add 2 0 19 plus 1 none gauss 10 1e-2 2"
    #"--pulse 40 add 2 0 19 plus 1 none gauss 30 1e-2 2"
    "--tmax 50"
    "-nosfml"
    "--output max"
    "--initRandom 0.5 random"
)


# Path to the program to launch
launch_program="./main_[FP].exe"

if [ "$makef" = true ]; then
    make clean
    make SFML=FALSE FP32=TRUE TARGET=${launch_program/\[FP\]/fp32} -j10
    make clean
    make SFML=FALSE TARGET=${launch_program/\[FP\]/fp64} -j10
fi

# Construct Folder
for fp in "fp32" "fp64"; do
    output_path_fp="$output_path$fp/"
    launch_program_fp="${launch_program/\[FP\]/$fp}"
    for N in "${N_list[@]}"; do
        output_path_N="$output_path_fp$N/"
        mkdir -p "$output_path_N"
        command=("$launch_program_fp" "${system_parameters[@]}" "--path" "$output_path_N" "--N" "$N")
        eval "${command[@]}" | tee -a "$output_path_N/output.log"
    done
done

# Define Python script as a multiline string
python_script='
import os
import matplotlib.pyplot as plt
import numpy as np

filepath = "'$output_path'"
print(filepath)

# Get Files
parents = ("fp32", "fp64")

N = len(parents)

data = {}
for parent in parents:
    parent_dir = os.path.join(filepath,parent)
    folders = sorted(os.listdir(parent_dir), key=lambda x: int(x))
    data[parent] = [[],[],[],[]] # X, Total Runtime, ms/ps, s/it
    for folder in folders:
        fp = os.path.join(parent_dir,folder,"output.log")
        tr,msps,sit = 0,0,0
        try:
            with open(fp, "r") as current:
                lines = [line for line in current.readlines() if "Total Runtime:" in line][0]
                information = lines.replace("Total Runtime: ","").replace("ms/ps -->","").replace("s/it","").replace("s","").split(" ")
                # Remove empty strings
                information = [x for x in information if len(x) > 0]
                print(information)
                tr,msps,sit = information
        except Exception as e:
            print(f"Error reading {fp}, error -> {e}")
        data[parent][0].append(int(folder))
        data[parent][1].append(float(tr))
        data[parent][2].append(float(msps))
        data[parent][3].append(float(sit))

fig,axes = plt.subplots(3,N)

axes = list(zip(*axes)) # Transpose axes
for ax,parent in zip(axes, parents):
    ax[0].set_title(parent)
    ax[0].plot(data[parent][0], data[parent][1], "o-")
    ax[0].text(0.5,0.95,f"Runtime",ha="center",va="top",transform=ax[0].transAxes)
    ax[1].plot(data[parent][0], data[parent][2], "o-")
    ax[1].text(0.5,0.95,"Time/ps",ha="center",va="top",transform=ax[1].transAxes)
    ax[2].plot(data[parent][0], data[parent][3], "o-")
    ax[2].text(0.5,0.95,"Time/it",ha="center",va="top",transform=ax[2].transAxes)
    [ax.set_xticklabels([]) for ax in ax[:-1]]
    [ax.set_yticks( np.linspace( np.min(v), np.max(v), 3 ) ) for ax,v in zip(ax, data[parent][1:])]
    [ax.yaxis.set_major_formatter(format) for ax,format in zip(ax, ["{x:.0f}s",lambda x,pos: f"{x*1e-3:.0f}s",lambda x,pos: f"{x*1e3:.0f}ms"])]


[ax.tick_params(axis="y", labelright=True, labelleft=False) for ax in axes[1]]

plt.tight_layout()

plt.savefig(os.path.join(filepath,"benchmark.pdf"))
'

# Call Python script
python -c "$python_script"
