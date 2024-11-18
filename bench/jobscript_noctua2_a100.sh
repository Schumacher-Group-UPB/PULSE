#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p dgx
#SBATCH -q devel
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBACTH -A pc2-mitarbeiter
#SBATCH -t 4:00:00

module reset
module load lang/Python/3.12.3-GCCcore-13.2.0
python3 bench.py -c noctua2_a100.json
