# Dynamic batching lab results

Local CPU check of a dynamic-batch queue in front of `SurgicalFeatureNet`. The model is an ONNX export (opset 18, input `input_frame` `[batch, 3, 128, 128]`, output `logits` `[batch, 4]`). ONNX Runtime runs the network. NVIDIA Triton Server is not running, and no GPU is used.

The queue budget is `max_queue_delay_microseconds: 3000` (3 ms). Preferred sizes are 4, 8, and 16. `max_batch_size` is 16. Each concurrency level sends 120 requests. The SLA line used here is P99 under 30 ms.

## Engine time only

`triton_client_bench.py` times `session.run` inside the worker. The clock does not include queue wait. Every call is batch size 1.

| Concurrency | P50 (ms) | P95 (ms) | P99 (ms) |
| --- | ---: | ---: | ---: |
| 1 | 0.25 | 0.45 | 0.72 |
| 4 | 0.68 | 1.34 | 3.54 |
| 8 | 1.00 | 1.78 | 2.31 |
| 16 | 1.45 | 3.60 | 6.37 |

## Queue without early preferred dispatch

The first scheduler waited out the delay unless the batch reached 16. On Windows, `queue.get(timeout=3ms)` also slept longer than 3 ms, so this table mixes that timer overshoot with queue delay.

| Concurrency | P50 (ms) | P95 (ms) | P99 (ms) | Mean batch | SLA |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 8.94 | 16.61 | 16.81 | 1.00 | PASS |
| 4 | 13.33 | 16.73 | 16.94 | 3.87 | PASS |
| 8 | 15.63 | 38.85 | 47.59 | 7.50 | FAIL |
| 16 | 7.11 | 12.20 | 13.94 | 13.33 | PASS |

Concurrency 8 failed because a batch of about 7 left one request for the next turn. That request paid the first batch's wait and compute, then its own wait and compute. Concurrency 16 filled toward the cap of 16 and dispatched sooner (mean batch 13.3, P99 13.9 ms).

## Early dispatch at a preferred size

The scheduler now sends the largest preferred size already in the queue (4, 8, or 16). If none of those sizes is reached, it waits only until a `perf_counter` deadline of 3 ms, then sends what it has. The deadline is not `queue.get(timeout=3ms)`.

| Concurrency | P50 (ms) | P95 (ms) | P99 (ms) | Mean batch | SLA |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 3.45 | 3.73 | 3.97 | 1.00 | PASS |
| 4 | 1.48 | 5.15 | 5.89 | 3.64 | PASS |
| 8 | 2.77 | 5.72 | 7.01 | 3.64 | PASS |
| 16 | 4.98 | 7.82 | 9.20 | 5.22 | PASS |

Concurrency 8 P99 moved from 47.59 ms to 7.01 ms. That is an 85% drop between these two scheduler runs on the same machine. Both the early-dispatch rule and the tighter 3 ms deadline changed between the runs, so the drop is not the early trigger alone.

Mean batch at concurrency 8 is 3.64, not 8. Size 4 is also preferred, so the queue leaves when 4 requests are ready. Concurrency 1 never reaches 4, so its P50 of 3.45 ms is the 3 ms budget plus a short forward. The highest P99 in this table is 9.20 ms.

Throughput was not recorded in these queue runs.

## What this does not show

This is a single-process mock of one batching thread. It does not measure Triton on a GPU, a second model instance, or a clinical system. A 30 ms line on this tiny CPU model is a lab budget, not a measurement from an operating room.
