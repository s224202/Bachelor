#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J testjob
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 6:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[gpu80gb]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s224202@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o HPC_logs/gpu_%J.out
#BSUB -e HPC_logs/gpu_%J.err
# -- end of LSF options --

nvidia-smi
module load cuda/11.6

Bachelor_env/bin/python3 Code/main.py