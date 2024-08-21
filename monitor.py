import psutil
import time
import os
import sys
import atexit
import json

if __name__ == '__main__':
    current_pid = os.getpid()
    monitored = sys.argv[1]
    output = sys.argv[2]
    output_file = os.path.join(output, 'memory_results.json')
    target_pid = None

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['cmdline'] and monitored in proc.info['cmdline'] and proc.pid != current_pid:
            proc.kill()
        
    while not target_pid:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['cmdline'] and monitored in proc.info['cmdline'] and proc.pid != current_pid:
                target_pid = proc.pid
                break

        if not target_pid:
            time.sleep(1)

    process = psutil.Process(target_pid)
    process_cmdline = process.cmdline()

    max_memory_stats = {
        "max_rss": 0,
        "max_vms": 0
    }

    def write_memory_max_to_json():
        with open(output_file, 'w') as report_file:
            json.dump(max_memory_stats, report_file, indent=2)

    atexit.register(write_memory_max_to_json)

    while 1:
        time.sleep(0.1)
        mem_info = process.memory_info()
        rss = mem_info.rss/1024**2
        vms = mem_info.vms/1024**2

        max_memory_stats["max_rss"] = max(max_memory_stats["max_rss"], rss)
        max_memory_stats["max_vms"] = max(max_memory_stats["max_vms"], vms)