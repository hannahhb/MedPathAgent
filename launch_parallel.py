import subprocess
import time
import os
import csv
import argparse

def launch_jobs(n_jobs, command_template, csv_path, total_rows, check_interval=60):
    # Launch processes
    procs = []
    slice_size = 1.0 / n_jobs
    for i in range(n_jobs):
        start = round(i * slice_size, 3)
        end = round((i + 1) * slice_size, 3)
        cmd = command_template.format(start=start, end=end)
        print(f"Launching job {i+1}/{n_jobs}: {cmd}")
        p = subprocess.Popen(cmd, shell=True)
        procs.append(p)
    start_time = time.time()

    # Monitor progress
    while True:
        try:
            rows_done = count_csv_rows(csv_path)
            elapsed = time.time() - start_time
            if rows_done > 0:
                rate = rows_done / elapsed
                remaining = total_rows - rows_done
                est_sec = remaining / rate if rate > 0 else float('inf')
            else:
                est_sec = float('inf')

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Rows done: {rows_done}/{total_rows}. Estimated remaining time: {format_duration(est_sec)}")
            if rows_done >= total_rows:
                print("Expected total reached. Breaking monitoring loop.")
                break
            time.sleep(check_interval)
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break

    # Wait for all jobs
    for p in procs:
        p.wait()
    print("All jobs completed.")

def count_csv_rows(csv_path):
    try:
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            # assume header row present
            return sum(1 for _ in reader) - 1
    except Exception as e:
        print("Error counting rows:", e)
        return 0

def format_duration(sec):
    if sec == float('inf'):
        return "unknown"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch split-range jobs and estimate time")
    parser.add_argument("--n_jobs", type=int, default=10, help="Number of jobs to launch")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV output")
    parser.add_argument("--total_rows", type=int, default=2491, help="Expected total rows upon completion")
    parser.add_argument("--check_interval", type=int, default=1, help="Seconds between checks")
    args = parser.parse_args()

    cmd_template = "python src/data_generation/generate_reasoning.py --dataset curebench --start {start} --end {end}"
    launch_jobs(args.n_jobs, cmd_template, args.csv_path, args.total_rows, args.check_interval)
