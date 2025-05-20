#!/usr/bin/env python3

import subprocess

def get_free_gpus():
    """
    Returns a list of strings specifying GPUs not in use.
    Note that this information is volatile.
    """
    gpus_smi = subprocess.check_output(["nvidia-smi", "--format=csv,noheader", "--query-gpu=index,gpu_bus_id"]).decode().strip()
    processes_smi = subprocess.check_output(["nvidia-smi", "--format=csv,noheader", "--query-compute-apps=pid,gpu_bus_id"]).decode().strip()

    gpus = {}
    for line in gpus_smi.split('\n'):
        idx, uuid = line.split(',')
        gpus[uuid.strip()] = idx.strip()

    # in case of no running processes
    if processes_smi == '':
        return list(gpus.values())

    used_gpus = set()
    for line in processes_smi.split('\n'):
        _, uuid = line.split(',')
        used_gpus.add(uuid.strip())

    free_gpus = list(set(gpus.keys()) - used_gpus)
    return sorted([gpus[free] for free in free_gpus])

if __name__ == "__main__":
    free_gpus = get_free_gpus()
    print(",".join(free_gpus))

