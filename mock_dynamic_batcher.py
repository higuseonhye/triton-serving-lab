import concurrent.futures
import os
import queue
import threading
import time

import numpy as np
import onnxruntime as ort

MODEL_PATH = "model_repository/surgical_feature_net/1/model.onnx"
MAX_QUEUE_DELAY_MS = 3.0  # config.pbtxt의 3000us에 해당
MAX_BATCH_SIZE = 16
PREFERRED_BATCH_SIZES = (4, 8, 16)


class DynamicBatchServer:
    def __init__(self, session, input_name):
        self.session = session
        self.input_name = input_name
        self.request_queue = queue.Queue()
        self.running = True
        self.batch_sizes = []
        self.worker_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.worker_thread.start()

    def _largest_preferred(self, count):
        fit = [size for size in PREFERRED_BATCH_SIZES if size <= count <= MAX_BATCH_SIZE]
        return max(fit) if fit else None

    def _collect_batch(self):
        """Wait for one request, then fill a preferred batch or the 3 ms budget.

        A preferred size already sitting in the queue is dispatched immediately.
        A smaller batch waits only until the deadline, then ships as-is.
        The deadline uses perf_counter because queue.get(timeout=3ms) on Windows
        often sleeps for a full timer tick.
        """
        try:
            first_req = self.request_queue.get(timeout=0.05)
        except queue.Empty:
            return None

        batch = [first_req]
        deadline = time.perf_counter() + (MAX_QUEUE_DELAY_MS / 1000.0)
        while len(batch) < MAX_BATCH_SIZE:
            available = min(MAX_BATCH_SIZE, len(batch) + self.request_queue.qsize())
            target = self._largest_preferred(available)
            if target is not None and len(batch) >= target:
                break
            if target is not None:
                try:
                    batch.append(self.request_queue.get_nowait())
                    continue
                except queue.Empty:
                    pass
            if time.perf_counter() >= deadline:
                break
            try:
                batch.append(self.request_queue.get_nowait())
            except queue.Empty:
                if self._largest_preferred(len(batch)) is not None:
                    break
                time.sleep(0)
        return batch

    def _batch_processor(self):
        while self.running:
            batch = self._collect_batch()
            if not batch:
                continue

            # 묶인 배치를 한 번에 ONNX 런타임으로 실행
            self.batch_sizes.append(len(batch))
            batched_input = np.concatenate([item["frame"] for item in batch], axis=0)
            _outputs = self.session.run(None, {self.input_name: batched_input})[0]

            # 결과를 개별 요청자에게 반환
            for item in batch:
                item["response_event"].set()

    def predict(self, dummy_frame):
        req_start = time.perf_counter()
        response_event = threading.Event()
        req_item = {
            "frame": dummy_frame,
            "response_event": response_event,
        }
        self.request_queue.put(req_item)
        response_event.wait()
        end = time.perf_counter()
        return (end - req_start) * 1000.0  # 총 End-to-End 레이턴시 (ms)


def run_benchmark(concurrency=8, num_requests=120):
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    server = DynamicBatchServer(session, input_name)

    dummy_frame = np.random.randn(1, 3, 128, 128).astype(np.float32)
    latencies = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(server.predict, dummy_frame) for _ in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            latencies.append(f.result())

    server.running = False
    latencies = np.array(latencies)
    return {
        "concurrency": concurrency,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_batch_size": float(np.mean(server.batch_sizes)) if server.batch_sizes else 0.0,
    }


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("=" * 78)
    print("Dynamic Batching End-to-End Latency Benchmark (Queue + Batching)")
    print("SLA Target: P99 < 30.0 ms | queue budget 3.0 ms | preferred batches 4, 8, 16")
    print("=" * 78)
    header = (
        f"{'Concurrency':<12} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | "
        f"{'P99 (ms)':<10} | {'Mean batch':<10} | {'SLA Status':<10}"
    )
    print(header)
    print("-" * 78)

    for c in [1, 4, 8, 16]:
        res = run_benchmark(concurrency=c, num_requests=120)
        sla = "PASS" if res["p99_ms"] < 30.0 else "FAIL"
        print(
            f"{res['concurrency']:<12} | {res['p50_ms']:<10.2f} | {res['p95_ms']:<10.2f} | "
            f"{res['p99_ms']:<10.2f} | {res['mean_batch_size']:<10.2f} | {sla:<10}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
