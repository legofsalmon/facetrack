# Model files

Every model here is committed, so a clone (or release ZIP) is complete
with no external downloads. If one is deleted or corrupted the launcher
restores it, or run:

```bash
.venv/bin/python -m yewee.doctor --fix
```

## What's here, and under what terms

| File | Used for | Source | Licence | Ships in a sold build? |
|---|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | face detection (default) | [opencv_zoo](https://github.com/opencv/opencv_zoo) | MIT | yes |
| `centerface_dynamic.onnx` | face detection (GPU tier) | [CenterFace](https://github.com/Star-Clouds/CenterFace) | MIT | yes |
| `emotion-ferplus-8.onnx` | expression estimation | [onnx/models](https://github.com/onnx/models) | MIT | yes |
| `human_segmentation_pphumanseg_2023mar.onnx` | people silhouette (fast) | [opencv_zoo](https://github.com/opencv/opencv_zoo) | Apache-2.0 | yes |
| `selfie_segmenter_landscape.onnx` | people silhouette (fast; crowd crops) | [MediaPipe selfie segmenter](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter) (landscape, 144x256), Google | Apache-2.0 | yes |
| `modnet_portrait.onnx` | people matte (quality) | [MODNet](https://github.com/ZHKKKe/MODNet) — repo states code **and models** are Apache-2.0 | Apache-2.0 | yes |
| `rvm_mobilenetv3_fp32.onnx` | people matte (best) | [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | **GPL-3.0** | **no — internal builds only** |

### Notes

- **`centerface_dynamic.onnx` is modified.** The upstream export fixes
  its input at 10x3x32x32, which onnxruntime cannot reshape. The shipped
  copy is the same weights with the batch/height/width dimensions marked
  dynamic, so the detector can run at any resolution (and benefit from a
  GPU at large input sizes). MIT permits this; the modification is
  recorded here for attribution.
- **`selfie_segmenter_landscape.onnx` is converted.** Google publishes
  the model as a TFLite file. It was converted with tf2onnx 1.17.0 (opset
  17), and its last layer, a MediaPipe custom op
  (`Convolution2DTransposeBias`) the converter cannot express, was
  rewritten as a standard `ConvTranspose` with the same weights and bias.
  Checked bit-exact against MediaPipe's own runtime on the same pixels
  (max difference 0.0000 on two test images). Apache-2.0 permits the
  conversion; it is recorded here for attribution.
- **RVM is deliberately excluded from distributed builds.** GPL-3.0
  would require shipping the whole product under GPL with source, which
  is incompatible with a paid closed-source licence. It stays available
  in internal builds — see `yewee/edition.py`.
- **SCRFD was removed** (was the GPU detector). InsightFace's models are
  licensed for non-commercial research only, so it could not ship in a
  product. CenterFace replaces it.
