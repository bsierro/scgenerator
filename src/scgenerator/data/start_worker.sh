#!/bin/bash
head /proc/cpuinfo
echo "starting ray worker node"
ray start --address $1 --redis-password=$2 --num-cpus=$SLURM_CPUS_PER_TASK
sleep infinity