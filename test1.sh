#!/bin/bash
#SBATCH -J MEMTEST
#SBATCH -p debug
#SBATCH --mem-per-cpu=2G
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o res.out
#SBATCH -e res.out
module unload compiler/rocm/2.9
module load apps/PyTorch/1.7-dynamic/hpcx-2.7.4-gcc-7.3.1-rocm3.9
python3 main.py