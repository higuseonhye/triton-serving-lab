import concurrent.futures
import os
import time

import numpy as np
import onnxruntime as ort

MODEL_PATH = "model_repository/surgical_feature_net/1/model.onnx"


def run_single_inference(session, input_name, dummy_frame):
    start = time.perf_counter()
    # Batch size 1 forward pass
    _ = session.run(None, {input_name: dummy_frame})
    end = time.perf_counter()
    return (end - start) * 1000.0  # ms 단위 변환


def benchmark_concurrency(session, input_name, concurrency, num_requests=100):
    dummy_frame = np.random.randn(1, 3, 128, 128).astype(np.float32)
    latencies = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_single_inference, session, input_name, dummy_frame)
            for _ in range(num_requests)
        ]
        for f in concurrent.futures.as_completed(futures):
            latencies.append(f.result())

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    throughput = num_requests / (np.sum(latencies) / 1000.0 / concurrency)

    return {
        "concurrency": concurrency,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "throughput_req_sec": throughput,
    }


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # ONNX Runtime Inference Session 생성
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(MODEL_PATH, options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print("=" * 65)
    print("Surgical Feature Net - Concurrency & Latency SLA Benchmark")
    print("SLA Target: P99 < 30.0 ms")
    print("=" * 65)
    print(f"{'Concurrency':<12} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'P99 (ms)':<10} | {'SLA Status':<10}")
    print("-" * 65)

    concurrency_levels = [1, 4, 8, 16]
    results = []

    for c in concurrency_levels:
        res = benchmark_concurrency(session, input_name, concurrency=c, num_requests=120)
        results.append(res)
        sla_pass = "PASS" if res["p99_ms"] < 30.0 else "FAIL"
        print(f"{res['concurrency']:<12} | {res['p50_ms']:<10.2f} | {res['p95_ms']:<10.2f} | {res['p99_ms']:<10.2f} | {sla_pass:<10}")

    print("=" * 65)


if __name__ == "__main__":
    main()
