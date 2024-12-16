#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --job-name="test"
#SBATCH --mem-per-cpu=1024

export OMP_NUM_THREADS=48;



# source $HOME/miniconda/bin/activate
# conda activate qmcmc

/cluster/home/zbinczyk/python/bin/python3 test.py

/cluster/home/zbinczyk/python/bin/python3 CL_Zofia_One_Class_test_file.py

#papermill ./time_dependency/optimal_t_vs_n.ipynb ./time_dependency/optimal_t_vs_n.ipynb

#CL_Zofia_One_Class.ipynb

# conda deactivate