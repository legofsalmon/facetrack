# Model files

The three ONNX models are committed to the repo, so a clone (or release
ZIP) is complete with no external downloads. If one is ever deleted or
corrupted, the launcher restores it automatically, or manually:

```bash
.venv/bin/python -m facetrack.doctor --fix
```

| File | Purpose | Size |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | face detector (CPU) | ~230 KB |
| `emotion-ferplus-8.onnx` | expression estimation | ~35 MB |
| `scrfd_10g.onnx` | face detector (NVIDIA GPU) | ~17 MB (pulled from a ~280 MB pack) |
