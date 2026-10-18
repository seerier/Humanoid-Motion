"""Resource evidence helpers for bounded compute-node experiments."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import time


@contextmanager
def monitor_resources(path: Path):
    """GPU-wide samples plus process Torch allocator peaks; no claim of attribution."""
    start = time.perf_counter()
    stop = threading.Event()
    report = {'gpu_samples': [], 'gpu_sampling_errors': [],
              'memory_scope': 'nvidia-smi is GPU-wide including other processes; torch is this process allocator only'}
    # Start system sampling before AppLauncher without importing Torch/CUDA first.
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()

    def sample() -> None:
        while not stop.is_set():
            try:
                selector = []  # Slurm GPU isolation exposes the assigned device(s); retain UUID evidence.
                output = subprocess.check_output(['nvidia-smi', *selector, '--query-gpu=uuid,memory.used,memory.total,utilization.gpu',
                                                   '--format=csv,noheader,nounits'], text=True, timeout=5)
                report['gpu_samples'].append({'seconds': time.perf_counter() - start, 'csv': output.strip()})
            except (OSError, subprocess.SubprocessError) as exc:
                if len(report['gpu_sampling_errors']) < 100:
                    report['gpu_sampling_errors'].append(str(exc))
            stop.wait(1)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield report
    finally:
        stop.set()
        worker.join(timeout=6)
        report['wall_seconds'] = time.perf_counter() - start
        peaks = {}
        for sample in report['gpu_samples']:
            for row in sample['csv'].splitlines():
                uuid, used, total, utilization = [item.strip() for item in row.split(',')]
                try:
                    peaks[uuid] = max(peaks.get(uuid, 0), float(used))
                except ValueError:
                    pass  # e.g. N/A under a driver/MIG configuration; retain raw evidence.
        report['sampled_gpu_peak_used_mib'] = peaks
        report['peak_sampling_interval_seconds'] = 1
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            report['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        path.write_text(json.dumps(report, indent=2) + '\n')
