# Model Artifacts

This directory stores metadata that can be committed and model binaries that
should stay local until Git LFS or release artifact storage is configured.

Committed:

- `grade_cnn_v1.metadata.json`
- `label_map.json`

Local-only:

- `grade_cnn_v1.pt`

Expected local checkpoint details:

```text
model_name: grade_cnn_v1
architecture: mobilenetv3_small_100
input_size: 224
size_bytes: 6248127
model_tier_used: cnn-v1
```

To verify the artifact:

```powershell
.venv\Scripts\python.exe -c "from app.main import health; print(health().model_loaded); print(health().model_bytes)"
```
