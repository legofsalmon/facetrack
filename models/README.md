# Model files

The three ONNX models live here but are not stored in git. The Setup
scripts download them automatically; to fetch them manually:

```bash
.venv/bin/python -m facetrack.doctor --fix
```

| File | Purpose | Size |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | face detector (CPU) | ~230 KB |
| `emotion-ferplus-8.onnx` | expression estimation | ~35 MB |
| `scrfd_10g.onnx` | face detector (NVIDIA GPU) | ~17 MB (pulled from a ~280 MB pack) |
