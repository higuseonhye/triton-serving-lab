# [Technical PRD] Low-latency serving scheduler for real-time vision ingestion

- **Version**: v1.0 (local prototype, queue scheduler checked)
- **Audience**: ingestion-queue design for an enterprise medical AI serving platform
- **Scope**: batch scheduling for a lightweight vision proxy. Measurements are a local CPU queue. They are not a clinical system and not a run of NVIDIA Triton Server.
- **Source tables**: [SERVING_RESULTS.md](SERVING_RESULTS.md)

## 1. Problem

A real-time clinical assistant would accept frames from more than one client. This prototype stands in for that load with thread pools at concurrency 1, 4, 8, and 16.

- **Batch size 1.** More in-flight calls take longer. Engine-only P99, which excludes queue wait, sat between 0.72 ms and 6.37 ms.
- **Dynamic batching that waits for the maximum batch.** The first scheduler held requests until 16 arrived or the delay expired. At concurrency 8 the mean batch was 7.5, and a request that missed the batch paid the previous cycle and then its own. P99 on that run was 47.59 ms. On Windows, `queue.get(timeout=3ms)` also slept longer than 3 ms, and that overshoot is inside the same table.

The goal is to dispatch at the largest preferred size already queued (4, 8, or 16), otherwise wait only until a 3.0 ms deadline, and keep **end-to-end P99 under 30.0 ms** on this local queue.

## 2. Shape of the pipeline

```
clients at concurrency 1, 4, 8, 16
        |
        v
dynamic batching queue
  - deadline: perf_counter() now + 3.0 ms
  - early exit: largest preferred size (4, 8, 16)
                that the queued requests can fill
        |
        v
np.concatenate -> (B, 3, 128, 128)
        |
        v
ONNX Runtime, CPU, opset 18, one instance
        |
        v
per-request event. Max P99 on this run: 9.20 ms
```

- **Model**: `SurgicalFeatureNet`. Input 128×128, four logits. It is not a model trained on surgical video.
- **Format**: ONNX. The export call asked for opset 17. PyTorch 2.13 saved opset 18. Axis 0 of `input_frame` and of `logits` is named `batch_size`.
- **Queue**: 3.0 ms budget, preferred sizes `[4, 8, 16]`, `max_batch_size` 16, one CPU instance.
- **Config file**: `model_repository/surgical_feature_net/config.pbtxt`. Triton did not load that file. The Python queue uses the same numbers.

The scheduler has three rules.

1. On the first request, set `deadline = now + 3.0 ms` and read it with `perf_counter`.
2. Let the available count be the in-hand batch plus requests already in the queue. If the in-hand batch has reached the largest preferred size that count can fill, run immediately. Do not spend the rest of the budget waiting for a larger preferred size that is not queued.
3. At the deadline, run whatever has arrived, as long as there is at least one request.

## 3. Measurements

Windows, Python 3.14, PyTorch 2.13, ONNX Runtime 1.25, CPU. The clock starts at `predict()` and stops when the response event is set. It includes compute, queue wait, and concatenation. It does not include network transport. Throughput was not recorded. Each cell is 120 requests.

Engine-only P99 at batch size 1, in concurrency order, was 0.72, 3.54, 2.31, and 6.37 ms.

| Concurrency | No early dispatch, P99 | Early dispatch + deadline fix, P99 | Improved P50 | Mean batch | SLA (< 30 ms) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 16.81 ms | 3.97 ms | 3.45 ms | 1.00 | PASS |
| 4 | 16.94 ms | 5.89 ms | 1.48 ms | 3.64 | PASS |
| 8 | 47.59 ms | 7.01 ms | 2.77 ms | 3.64 | PASS |
| 16 | 13.94 ms | 9.20 ms | 4.98 ms | 5.22 | PASS |

Concurrency 8 P99 moved from 47.59 ms to 7.01 ms, an 85.3% drop. Two things changed between the runs: early dispatch at a preferred size, and a `perf_counter` deadline in place of `queue.get(timeout=3ms)`. The drop is not the early trigger by itself.

Mean batch 3.64 at concurrency 8 means the queue left at size 4. It did not wait for 8. P50 3.45 ms at concurrency 1 is the 3.0 ms budget, because a single request never fills a preferred size. The longest P99 in this table is 9.20 ms.

## 4. Later work and acceptance

The items below are not done.

1. Load the same numbers through Triton's dynamic batcher.
2. Measure latency and throughput together at concurrency 1–32 with `perf_analyzer`.
3. Compare `instance_group.count` 1 and 2.

Acceptance:

- [x] Opset 18 ONNX checked for a dynamic batch axis and for the input and output names.
- [x] On this local queue, P99 at concurrency 1–16 stayed under 30.0 ms. The max was 9.20 ms.
- [ ] End-to-end P99 under 30.0 ms through Triton gRPC, including transport. Not measured.
- [ ] Saturation throughput. The figure 500 req/s is a target for that later run. It is not a result of this experiment.

## 5. Summary

When concurrent vision requests share one queue, tail latency piles up across batch boundaries.

On a local CPU, a Python queue in front of ONNX Runtime was timed against a 30 ms budget. The scheduler that waited for batch 16 recorded P99 47.59 ms at concurrency 8. Requests that missed a batch paid two cycles, and the Windows wait slept past 3 ms.

The revised scheduler dispatches the largest preferred size already possible among 4, 8, and 16, and it closes the wait with `perf_counter`. Concurrency 8 P99 was 7.01 ms. Concurrency 16 P99 was 9.20 ms.

The next check is Triton's C++ dynamic batcher and `perf_analyzer`. This document stops at the local queue.
