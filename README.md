# triton-serving-lab

Local queue in front of a small ONNX vision model. It checks a 3 ms batching budget and preferred sizes 4, 8, and 16. NVIDIA Triton Server is not started.

| Document | |
| --- | --- |
| [PRD.ko.md](PRD.ko.md) | Korean technical PRD |
| [PRD.en.md](PRD.en.md) | English technical PRD |
| [SERVING_RESULTS.md](SERVING_RESULTS.md) | Tables from the CPU runs |

## Run

```bash
python -m pip install onnx onnxscript onnxruntime numpy
python export_onnx.py
python triton_client_bench.py
python mock_dynamic_batcher.py
```

`export_onnx.py` writes `model_repository/surgical_feature_net/1/model.onnx`. On PyTorch 2.13 the saved opset is 18. `config.pbtxt` is the Triton-shaped config. The benchmark scripts call ONNX Runtime directly.
