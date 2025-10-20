# Copyright (c) 2025 Zeyuan Li Wuhan University. Licensed under the MIT License.
# See the LICENSE file in the repository root for full details.
import json
import os
import re
import time
import subprocess
import shutil

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def ssh_command_with_sshpass(user, host, port, password, command):
    cmd = ["sshpass", "-p", password, "ssh", "-p", str(port), f"{user}@{host}", command]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def rsync_to_remote(user, host, port, password, local_path, remote_path):
    cmd = [
        "sshpass", "-p", password, "rsync", "-avz",
        "-e", f"ssh -p {port}",
        local_path,
        f"{user}@{host}:{remote_path}"
    ]
    subprocess.run(cmd, check=True)

def rsync_from_remote(user, host, port, password, remote_path, local_path):
    cmd = [
        "sshpass", "-p", password, "rsync", "-avz",
        "-e", f"ssh -p {port}",
        f"{user}@{host}:{remote_path}",
        local_path
    ]
    subprocess.run(cmd, check=True)

def remote_mkdir(user, host, port, password, remote_dir):
    command = f"mkdir -p {remote_dir}"
    stdout, stderr, rc = ssh_command_with_sshpass(user, host, port, password, command)
    if rc != 0:
        raise RuntimeError(f"Remote directory creation failed: {stderr}")

def submit_job_and_get_id_remote(user, host, port, password, remote_subdir, sub_vasp_file):
    # Submit job remotely
    command = f"cd {remote_subdir} && sbatch {sub_vasp_file}"
    stdout, stderr, rc = ssh_command_with_sshpass(user, host, port, password, command)
    if rc == 0:
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if match:
            job_id = match.group(1)
            return job_id
        else:
            print(f"Remote submission for {remote_subdir} failed, no Job ID matched. Output: {stdout}")
            return None
    else:
        print(f"Error submitting job for {remote_subdir}: {stderr}")
        return None

def is_job_completed_remote(user, host, port, password, job_id):
    command = f"squeue -j {job_id}"
    stdout, stderr, rc = ssh_command_with_sshpass(user, host, port, password, command)
    return job_id not in stdout

def check_vasp_task_completed(task_dir, check_type='simple'):
    if not os.path.exists(task_dir):
        return False
    if check_type == 'simple':
        # Consider completed if OUTCAR file exists in task directory
        return 'OUTCAR' in os.listdir(task_dir)
    return False

def load_record_file(record_file):
    # record.txt format: POSCAR_name copied cluster_name job_id completed
    # If file doesn't exist, create empty file and return empty dictionary
    records = {}
    if not os.path.exists(record_file):
        return records
    with open(record_file, 'r') as rf:
        lines = rf.readlines()
    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        poscar_name, copied, cluster_name, job_id, completed = line.split('\t')
        records[poscar_name] = {
            'copied': int(copied),
            'cluster_name': cluster_name,
            'job_id': job_id if job_id != 'N/A' else None,
            'completed': int(completed)
        }
    return records

def save_record_file(record_file, records):
    # Rewrite record file
    # POSCAR_name copied cluster_name job_id completed
    with open(record_file, 'w') as rf:
        rf.write("POSCAR_name\tcopied\tcluster_name\tjob_id\tcompleted\n")
        for poscar_name, info in records.items():
            job_id_str = info['job_id'] if info['job_id'] else 'N/A'
            rf.write(f"{poscar_name}\t{info['copied']}\t{info['cluster_name']}\t{job_id_str}\t{info['completed']}\n")

def get_unfinished_poscars(poscar_files, records):
    # Unfinished POSCAR definition: completed=0
    # Uncopied POSCAR: copied=0 means folder hasn't been created
    # Return all POSCARs with completed=0 (i.e., POSCARs that still need calculation)
    unfinished = []
    for p in poscar_files:
        if p not in records:
            # No record means unfinished and uncopied
            unfinished.append(p)
        else:
            if records[p]['completed'] == 0:
                # Unfinished needs reassignment
                # But if copied=1 and job_id not completed yet, need to wait.
                # However, unfinished here only includes all not finally completed
                unfinished.append(p)
    return unfinished

def all_tasks_completed(records, poscar_files):
    # All POSCARs completed if all have completed=1
    for p in poscar_files:
        if p not in records or records[p]['completed'] == 0:
            return False
    return True

def get_current_running_jobs(records):
    # Return all running job_ids (completed=0 and job_id not empty)
    running_jobs = []
    for p, info in records.items():
        if info['completed'] == 0 and info['job_id'] is not None:
            running_jobs.append((p, info['cluster_name'], info['job_id']))
    return running_jobs

def count_tasks_per_node(records, nodes):
    # Count current running tasks per node
    counts = {node['cluster_name']:0 for node in nodes}
    for p, info in records.items():
        if info['completed'] == 0 and info['job_id'] is not None:
            if info['cluster_name'] in counts:
                counts[info['cluster_name']] += 1
    return counts

def create_and_submit_task(poscar_name, node, config, records, log_file):
    # Create corresponding task folder locally
    # Local structure: work/cluster_name/vasp_POSCAR_xxx
    cluster_name = node['cluster_name']
    local_work_dir = node['local_work_dir'] # Usually same as global, but can be different
    poscar_source_dir = config['poscar_source_dir']
    files_to_copy = config['files_to_copy']
    sub_vasp_file = node['sub_vasp_file']

    node_local_dir = os.path.join(local_work_dir, cluster_name)
    if not os.path.exists(node_local_dir):
        os.makedirs(node_local_dir, exist_ok=True)

    task_dir_name = "vasp_" + poscar_name
    task_dir_path = os.path.join(node_local_dir, task_dir_name)
    if not os.path.exists(task_dir_path):
        os.makedirs(task_dir_path)
    
    # Copy POSCAR
    src_poscar = os.path.join(poscar_source_dir, poscar_name)
    dst_poscar = os.path.join(task_dir_path, "POSCAR")
    shutil.copy(src_poscar, dst_poscar)

    # Copy other files
    for fname in files_to_copy:
        src_file = os.path.join(node['local_work_dir'], fname)
        if not os.path.exists(src_file):
            with open(log_file, 'a') as lf:
                lf.write(f"Warning: File {fname} does not exist in local work directory, cannot copy to {task_dir_path}.\n")
        else:
            shutil.copy(src_file, task_dir_path)

    # Copy corresponding node's sub_vasp_file
    src_sub_file = os.path.join(node['local_work_dir'], sub_vasp_file)
    if not os.path.exists(src_sub_file):
        with open(log_file, 'a') as lf:
            lf.write(f"Warning: File {sub_vasp_file} does not exist\n")
    else:
        shutil.copy(src_sub_file, os.path.join(task_dir_path, sub_vasp_file))

    # Sync to remote
    remote_subdir = os.path.join(node['remote_work_dir'], task_dir_name)
    remote_mkdir(node['user'], node['host'], node['port'], node['password'], remote_subdir)
    rsync_to_remote(node['user'], node['host'], node['port'], node['password'], task_dir_path + "/", remote_subdir + "/")

    # Submit job
    job_id = submit_job_and_get_id_remote(node['user'], node['host'], node['port'], node['password'], remote_subdir, sub_vasp_file)
    if job_id:
        records[poscar_name] = {
            'copied': 1,
            'cluster_name': cluster_name,
            'job_id': job_id,
            'completed': 0
        }
        with open(log_file, 'a') as lf:
            lf.write(f"{poscar_name}: Successfully submitted to {cluster_name}, Job ID = {job_id}\n")
    else:
        # Submission failed
        records[poscar_name] = {
            'copied': 1,
            'cluster_name': cluster_name,
            'job_id': None,
            'completed': 0
        }
        with open(log_file, 'a') as lf:
            lf.write(f"{poscar_name}: Failed to submit to {cluster_name}\n")

if __name__ == "__main__":
    # Load configuration
    config = load_config("machine.json")
    nodes = config["nodes"]
    poscar_source_dir = config["poscar_source_dir"]

    # Global records and logs
    # Store logs and records in global work directory
    global_work_dir = nodes[0]['local_work_dir']  # Assume all nodes have same local_work_dir
    log_file = os.path.join(global_work_dir, "process.log")
    record_file = os.path.join(global_work_dir, "record.txt")

    poscar_files = [f for f in os.listdir(poscar_source_dir) if f.startswith("POSCAR")]
    poscar_files.sort()  # Sort for consistent assignment order

    # Initialize or load records
    records = load_record_file(record_file)
    if not records:
        # Initialize record file
        with open(record_file, 'w') as rf:
            rf.write("POSCAR_name\tcopied\tcluster_name\tjob_id\tcompleted\n")
        # Initialize all POSCAR status
        for p in poscar_files:
            if p not in records:
                records[p] = {'copied':0, 'cluster_name':'N/A', 'job_id':None, 'completed':0}

    with open(log_file, 'a') as lf:
        lf.write("Starting task allocation and submission...\n")

    # Main loop
    while not all_tasks_completed(records, poscar_files):
        # Get unfinished poscars
        unfinished = get_unfinished_poscars(poscar_files, records)

        # Allocate nodes for uncopied (copied=0) and unfinished POSCARs
        # First check current running tasks per node, compare with max_tasks
        counts = count_tasks_per_node(records, nodes)

        # Try to assign tasks to nodes with available slots
        tasks_assigned = 0
        for p in unfinished:
            if records[p]['copied'] == 0:
                # Find a node with free slot
                assigned = False
                for node in nodes:
                    cluster_name = node['cluster_name']
                    if counts[cluster_name] < node['max_tasks']:
                        # Assign this task to the node
                        create_and_submit_task(p, node, config, records, log_file)
                        # Update count
                        counts[cluster_name] += 1
                        tasks_assigned += 1
                        assigned = True
                        break
                if not assigned:
                    # No node has available resources
                    break

        # Save records
        save_record_file(record_file, records)

        if tasks_assigned == 0:
            # No new tasks assigned, meaning all nodes are full or no new poscars to assign
            # Wait for a task to complete
            running_jobs = get_current_running_jobs(records)
            if not running_jobs:
                # No running tasks and no tasks to assign, might all be completed
                continue

            # Wait for task completion
            with open(log_file, 'a') as lf:
                lf.write("All nodes at full capacity, waiting for task completion...\n")

            # Poll until a task completes
            while True:
                running_jobs = get_current_running_jobs(records)
                if not running_jobs:
                    # All completed
                    break
                # Check task status
                any_finished = False
                for p, c_name, j_id in running_jobs:
                    node = [n for n in nodes if n['cluster_name'] == c_name][0]
                    if is_job_completed_remote(node['user'], node['host'], node['port'], node['password'], j_id):
                        # Sync results back to local and check OUTCAR
                        task_dir_name = "vasp_" + p
                        node_local_dir = os.path.join(node['local_work_dir'], c_name)
                        task_dir_path = os.path.join(node_local_dir, task_dir_name)
                        remote_subdir = os.path.join(node['remote_work_dir'], task_dir_name)
                        rsync_from_remote(node['user'], node['host'], node['port'], node['password'], remote_subdir + "/", task_dir_path + "/")

                        completed_flag = 1 if check_vasp_task_completed(task_dir_path) else 0
                        records[p]['completed'] = completed_flag
                        records[p]['job_id'] = j_id  # Keep unchanged
                        # After one task ends, can release slot for next task
                        any_finished = True
                        with open(log_file, 'a') as lf:
                            lf.write(f"Task {p} completed on node {c_name}, completed={completed_flag}\n")

                if any_finished:
                    # Some tasks completed, update records and break waiting loop
                    save_record_file(record_file, records)
                    break
                else:
                    # No completed tasks, wait some time before checking again
                    time.sleep(10)
        else:
            # Successfully assigned tasks, continue to next round
            # Can wait a bit before checking task completion
            time.sleep(5)

    with open(log_file, 'a') as lf:
        lf.write("All POSCAR calculations completed.\n")
