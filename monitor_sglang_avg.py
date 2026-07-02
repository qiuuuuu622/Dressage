#!/usr/bin/env python3
import re
import subprocess
import time
from datetime import datetime
from urllib.request import urlopen

URL = "http://10.244.244.180:8000/engine_metrics"
INTERVAL = 30
OUT = "/root/Dressage/sglang_4engine_avg_metrics.log"


def read_metrics():
    return urlopen(URL, timeout=5).read().decode()


def parse_samples(text, metric):
    vals = {}
    pattern = re.compile(rf"^{re.escape(metric)}\{{([^}}]*)\}}\s+([-+0-9.eE]+)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        labels, val = match.groups()
        worker = re.search(r'worker_addr="([^"]+)"', labels)
        if worker:
            vals[worker.group(1)] = float(val)
    return vals


def parse_counter_by_mode(text, metric, mode):
    vals = {}
    pattern = re.compile(rf"^{re.escape(metric)}\{{([^}}]*)\}}\s+([-+0-9.eE]+)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        labels, val = match.groups()
        if f'mode="{mode}"' not in labels:
            continue
        worker = re.search(r'worker_addr="([^"]+)"', labels)
        if worker:
            vals[worker.group(1)] = float(val)
    return vals


def avg(values):
    return sum(values) / len(values) if values else 0.0


def gpu_stats():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    rows = []
    for line in output.strip().splitlines():
        idx, mem, util, power = [part.strip() for part in line.split(",")]
        rows.append((int(idx), float(mem), float(util), float(power)))
    return rows


def main():
    previous = {}
    metric_names = [
        "sglang_cache_hit_rate",
        "sglang_num_running_reqs",
        "sglang_num_queue_reqs",
        "sglang_e2e_request_latency_seconds_sum",
        "sglang_e2e_request_latency_seconds_count",
        "sglang_prompt_tokens_total",
        "sglang_generation_tokens_total",
    ]

    with open(OUT, "a", buffering=1) as log:
        log.write(f"# start {datetime.now().isoformat()} interval={INTERVAL}s url={URL}\n")
        log.write(
            "# ts engines gpu_util_avg gpu_mem_avg_mib gpu_power_avg_w "
            "cache_hit_avg running_req_avg waiting_req_avg decode_graph_pass_delta_s "
            "prefill_pass_delta_s e2e_count_delta_s e2e_latency_avg_s "
            "output_tokens_delta_s input_tokens_delta_s\n"
        )
        while True:
            now = time.time()
            ts = datetime.now().isoformat(timespec="seconds")
            try:
                text = read_metrics()
                gpu = gpu_stats()

                workers = set()
                metrics = {}
                for name in metric_names:
                    metrics[name] = parse_samples(text, name)
                    workers.update(metrics[name])

                decode = parse_counter_by_mode(
                    text, "sglang_cuda_graph_passes_total", "decode_cuda_graph"
                )
                prefill = parse_counter_by_mode(
                    text, "sglang_cuda_graph_passes_total", "prefill_none"
                )
                workers.update(decode)
                workers.update(prefill)

                totals = {name: sum(values.values()) for name, values in metrics.items()}
                totals["decode"] = sum(decode.values())
                totals["prefill"] = sum(prefill.values())

                dt = max(now - previous.get("time", now), 1e-6)

                def rate(name):
                    if name not in previous:
                        return 0.0
                    return (totals.get(name, 0.0) - previous.get(name, 0.0)) / dt

                e2e_count = totals.get("sglang_e2e_request_latency_seconds_count", 0.0)
                e2e_sum = totals.get("sglang_e2e_request_latency_seconds_sum", 0.0)
                prev_count = previous.get("sglang_e2e_request_latency_seconds_count", e2e_count)
                prev_sum = previous.get("sglang_e2e_request_latency_seconds_sum", e2e_sum)
                delta_count = e2e_count - prev_count
                delta_sum = e2e_sum - prev_sum
                e2e_avg = delta_sum / delta_count if delta_count > 0 else 0.0

                log.write(
                    f"{ts} engines={len(workers)} "
                    f"gpu_util_avg={avg([row[2] for row in gpu]):.1f} "
                    f"gpu_mem_avg_mib={avg([row[1] for row in gpu]):.0f} "
                    f"gpu_power_avg_w={avg([row[3] for row in gpu]):.1f} "
                    f"cache_hit_avg={avg(list(metrics['sglang_cache_hit_rate'].values())):.4f} "
                    f"running_req_avg={avg(list(metrics['sglang_num_running_reqs'].values())):.2f} "
                    f"waiting_req_avg={avg(list(metrics['sglang_num_queue_reqs'].values())):.2f} "
                    f"decode_graph_pass_delta_s={rate('decode'):.2f} "
                    f"prefill_pass_delta_s={rate('prefill'):.2f} "
                    f"e2e_count_delta_s={rate('sglang_e2e_request_latency_seconds_count'):.2f} "
                    f"e2e_latency_avg_s={e2e_avg:.2f} "
                    f"output_tokens_delta_s={rate('sglang_generation_tokens_total'):.2f} "
                    f"input_tokens_delta_s={rate('sglang_prompt_tokens_total'):.2f}\n"
                )
                previous = {"time": now, **totals}
            except Exception as exc:
                log.write(f"{ts} ERROR {exc!r}\n")
            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
