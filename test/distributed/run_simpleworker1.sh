#!/bin/sh

python simple_worker1.py --job_name 'ps' --task_index 0 & 
python simple_worker1.py --job_name 'worker' --task_index 0 & 
python simple_worker1.py --job_name 'worker' --task_index 1 & 
