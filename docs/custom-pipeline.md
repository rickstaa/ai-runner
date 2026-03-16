# Creating a Custom Pipeline

This guide explains how to create a custom pipeline for the AI Runner from a **separate repository**. We'll use [scope-runner](https://github.com/livepeer/scope-runner) as a reference implementation.

> **Note**: The current integration process requires small modifications to both `ai-runner` and `go-livepeer` repositories. We're working toward a more generic plugin architecture that would eliminate these manual changes, but for now this guide documents the current workflow.

## Overview

A custom pipeline is a Python package that:

1. Extends the `ai-runner[realtime]` library as a dependency (or `ai-runner[batch]` for a batch pipeline)
2. Uses the [`@pipeline`](../runner/src/runner/live/pipelines/create.py) decorator to define frame processing logic
3. Optionally defines custom parameters extending [`BaseParams`](../runner/src/runner/live/pipelines/interface.py#L10)
4. Provides a `prepare_models()` classmethod for model download/compilation
5. Ships as a Docker image, ideally extending `livepeer/ai-runner:live-base`

## Prerequisites

- Python 3.10+ (stricter dependency will likely come from your pipeline code)
- [uv](https://docs.astral.sh/uv/) package manager
- Docker with NVIDIA GPU support
- Access to a CUDA-capable GPU for testing

---

## Step 1: Create a New Project with uv

Initialize a new Python project:

```bash
mkdir my-pipeline
cd my-pipeline
uv init --lib
```

Configure your `pyproject.toml`:

```toml
[project]
name = "my-pipeline"
version = "0.1.0"
requires-python = ">=3.10.12,<3.11"
dependencies = [
    "ai-runner[realtime]",
    # Add your pipeline-specific dependencies here
    # "torch>=2.0.0",
    # "transformers",
]

[project.scripts]
my-pipeline = "my_pipeline.main:main"

[tool.uv.sources]
# Pin to a specific ai-runner release for reproducibility
ai-runner = { git = "https://github.com/livepeer/ai-runner.git", rev = "v0.14.0", subdirectory = "runner" }

[tool.uv]
package = true

[tool.setuptools.packages.find]
where = ["src"]
```

Create the project structure:

```bash
mkdir -p src/my_pipeline/pipeline
touch src/my_pipeline/__init__.py
touch src/my_pipeline/main.py
touch src/my_pipeline/pipeline/__init__.py
touch src/my_pipeline/pipeline/pipeline.py
touch src/my_pipeline/pipeline/params.py
```

---

## Step 2: Implement the Pipeline

Use the `@pipeline` decorator to define your pipeline. The decorator handles frame queues, lifecycle management, parameter validation, and threading automatically.

**Function form** — simplest possible pipeline:

```python
# src/my_pipeline/pipeline/pipeline.py
import torch
from runner.live.pipelines import pipeline, BaseParams
from runner.live.trickle import VideoFrame

@pipeline(name="green-shift")
async def green_shift(frame: VideoFrame, params: BaseParams) -> torch.Tensor:
    # Process frame tensor and return modified tensor
    tensor = frame.tensor.clone()
    tensor[:, :, :, 1] = torch.clamp(tensor[:, :, :, 1] + 0.3, -1.0, 1.0)
    return tensor
```

**Class form** — for state, device setup, and mid-stream parameter updates:

```python
# src/my_pipeline/pipeline/pipeline.py
import logging
import torch
import torch.nn.functional as F
from pydantic import Field
from runner.live.pipelines import pipeline, BaseParams
from runner.live.trickle import VideoFrame

class EdgeParams(BaseParams):
    threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Edge threshold.")
    colorize: bool = Field(default=False, description="Colorize edges by direction.")

@pipeline(name="edge-detect", params=EdgeParams)
class EdgeDetect:

    def on_ready(self, **params):
        self._threshold = params.get("threshold", 0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)

    def transform(self, frame: VideoFrame, params: EdgeParams) -> torch.Tensor:
        tensor = frame.tensor.to(self.device)
        gray = tensor.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        edges_x = F.conv2d(gray, self.sobel_x, padding=1)
        edges_y = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        magnitude = magnitude / (magnitude.max() + 1e-8)
        edges = (magnitude > self._threshold).float()
        out = edges.expand(-1, 3, -1, -1).permute(0, 2, 3, 1)
        return (out * 2.0 - 1.0)

    def on_update(self, **params):
        self._threshold = params.get("threshold", 0.1)
        logging.info(f"Edge threshold updated: {self._threshold}")

    def on_stop(self):
        logging.info("EdgeDetect stopped")
```

**Lifecycle methods:**

| Method | Required | When it runs | What to do here |
|---|---|---|---|
| `prepare_models(cls)` | No | **Build time** | Download weights, compile TensorRT engines |
| `on_ready(self, **params)` | No | **Process startup** | Load model from disk to GPU |
| `transform(self, frame, params)` | Yes | **Every frame** | Run inference, return tensor |
| `on_update(self, **params)` | No | **Mid-stream** | Handle param changes |
| `on_stop(self)` | No | **Shutdown** | Release resources |

Both `async def` and `def` work for all methods. Sync functions automatically run in a thread pool.

See [`examples/live-video-to-video/`](../examples/live-video-to-video/) for complete working examples.

### Define Parameters (Optional)

```python
# src/my_pipeline/pipeline/params.py
from runner.live.pipelines import BaseParams

class MyPipelineParams(BaseParams):
    # Define your custom fields here
```

### Keep Module Exports Minimal

> **⚠️ Important**: Do **not** export `Pipeline` or `Params` classes from `__init__.py`. The loader imports these by their full path (`module.path:ClassName`), and re-exporting from `__init__.py` would trigger expensive imports (torch, etc.) when only loading the params class.

Keep `src/my_pipeline/pipeline/__init__.py` empty or minimal with only basic exports.

---

## Step 3: Create the Application Entrypoint

Create `src/my_pipeline/main.py`:

```python
import os
from pathlib import Path

from runner.app import start_app
from runner.live.pipelines import PipelineSpec

pipeline_spec = PipelineSpec(
    name="my-pipeline",  # Must match `model_id` in go-livepeer
    pipeline_cls="my_pipeline.pipeline.pipeline:MyPipeline",
    params_cls="my_pipeline.pipeline.params:MyPipelineParams",
    initial_params={
        "prompt": "default prompt"
    },
)

if __name__ == "__main__":
    start_app(pipeline=pipeline_spec)
```

---

## Step 4: Write the Dockerfile

Create a `Dockerfile` in your project root:

```dockerfile
# Use the ai-runner live-base image
# Pin to a specific version for reproducibility
ARG BASE_IMAGE=livepeer/ai-runner:live-base-57efd92
FROM ${BASE_IMAGE}

# Install system dependencies if needed

WORKDIR /app


# Install only dependencies first (optimal layer caching)
COPY pyproject.toml uv.lock ./
RUN mkdir -p src/my_pipeline/pipeline && \
    touch src/my_pipeline/__init__.py && \
    touch src/my_pipeline/pipeline/__init__.py
RUN uv sync --locked --no-install-project

# Actually install the whole project now
COPY src/my_pipeline/ ./src/my_pipeline/
RUN uv sync --locked

# Disable HuggingFace Hub online access at runtime
# During models download/prepare, this is automatically disabled by dl_checkpoints.sh
ENV HF_HUB_OFFLINE=1

# Version metadata
ARG GIT_SHA
ARG VERSION="undefined"
ENV GIT_SHA="${GIT_SHA}" \
    VERSION="${VERSION}"

CMD ["uv", "run", "--frozen", "my-pipeline"]
```

---

## Step 5: Implement Model Preparation

The `prepare_models()` classmethod runs at **build time** when an operator sets up their node, not when a stream or request arrives. It is triggered by the `PREPARE_MODELS=1` environment variable (or `--prepare-models` flag), and is called automatically by `dl_checkpoints.sh` during operator setup.

This is the right place for any expensive one-time work:

- **Downloading model weights** from HuggingFace, Google Drive, etc.
- **Compiling TensorRT engines** for optimized GPU inference
- **Converting model formats** (e.g., ONNX export, quantization)
- **Warming up caches** or generating lookup tables

Unlike runtime (where `HF_HUB_OFFLINE=1` prevents accidental downloads), `prepare_models` runs with full network access so you can fetch weights from HuggingFace, Google Drive, or other sources.

```python
@classmethod
def prepare_models(cls):
    """Download and prepare all required models."""
    import logging
    from huggingface_hub import snapshot_download

    models_dir = Path(os.environ.get("MODEL_DIR", "/models")) / "MyPipeline--models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Downloading models to {models_dir}")

    # Download from HuggingFace
    snapshot_download(
        "my-org/my-model",
        local_dir=models_dir / "my-model",
        local_dir_use_symlinks=False,
    )

    # Optional: compile TensorRT engine for faster inference
    # import torch_tensorrt
    # model = torch.load(models_dir / "my-model" / "model.pt")
    # trt_model = torch_tensorrt.compile(model, inputs=[...])
    # torch.save(trt_model, models_dir / "my-model" / "model_trt.pt")

    logging.info("Model preparation complete")
```

Then in `on_ready`, just load the pre-downloaded (and optionally pre-compiled) model from disk:

```python
def on_ready(self, **params):
    """Load model from disk to GPU. Should be fast (seconds, not minutes)."""
    models_dir = Path(os.environ.get("MODEL_DIR", "/models")) / "MyPipeline--models"
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = torch.load(models_dir / "my-model" / "model.pt", map_location=self.device)
    self.model.eval()
```

---

## Step 6: Integration with Livepeer Infrastructure

> **⚠️ Current Limitation**: These manual changes are required because the pipeline registry isn't fully dynamic yet. We intend to make this configurable without code changes in the base projects in the future (pull requests welcome!)

### 6.1 Update `ai-runner/runner/dl_checkpoints.sh`

Add your pipeline to the download script so operators can prepare models:

```bash
# In dl_checkpoints.sh, add your pipeline image variable
AI_RUNNER_MY_PIPELINE_IMAGE=${AI_RUNNER_MY_PIPELINE_IMAGE:-my-org/my-pipeline}

# Add to download_live_models() case statement
function download_live_models() {
  case "$PIPELINE" in
  # ... existing cases ...
  "my-pipeline")
    printf "\nPreparing my-pipeline models...\n"
    prepare_my_pipeline_models
    ;;
  "all")
    # ... existing code ...
    prepare_my_pipeline_models
    ;;
  esac
}

# Add preparation function
function prepare_my_pipeline_models() {
  printf "\nPreparing my-pipeline models...\n"
  run_pipeline_prepare "my-pipeline" "$AI_RUNNER_MY_PIPELINE_IMAGE"
}
```

### 6.2 Update `go-livepeer/ai/worker/docker.go`

Add your pipeline to the container image mapping:

```go
// In livePipelineToImage map
var livePipelineToImage = map[string]string{
    // ... existing entries ...
    "my-pipeline": "my-org/my-pipeline",
}
```

This allows the orchestrator to start containers for your pipeline when jobs are received.

### 6.3 Register with Orchestrator

When running the orchestrator, configure it to advertise your pipeline capability through the `aiModels.json` configuration.

---

## Testing Your Pipeline

### Local Testing

1. **Install dependencies**:

   ```bash
   uv sync
   ```

2. **Prepare models** (download/compile):

   ```bash
   uv run my-pipeline --prepare-models
   ```

3. **Run the pipeline**:

   ```bash
   uv run my-pipeline
   ```

4. **Test with the health endpoint**:

   ```bash
   curl http://localhost:8000/health
   ```

### Integration Testing with go-livepeer Box

For full end-to-end testing with the Livepeer stack (gateway, orchestrator, Trickle streams), use the [go-livepeer box](https://github.com/livepeer/go-livepeer/blob/master/box/box.md) with your local runner.

1. **Start your local pipeline**:

   ```bash
   uv run my-pipeline
   # Pipeline starts on http://localhost:8000
   ```

2. **Create an `aiModels.json` file** pointing to your local runner:

   ```json
   [
     {
       "pipeline": "live-video-to-video",
       "model_id": "my-pipeline",
       "url": "http://localhost:8000"
     }
   ]
   ```

   The `url` field tells the orchestrator to use your local runner instead of starting a Docker container. The `model_id` must match your pipeline's `name` in the `PipelineSpec`.

3. **Start the go-livepeer box** with your config:

   ```bash
   cd /path/to/go-livepeer/box

   # Point to your aiModels.json file
   export AI_MODELS_JSON=/path/to/aiModels.json

   # Start the orchestrator and gateway
   make box
   ```

4. **Stream and playback**:

   ```bash
   make box-stream    # Start streaming
   make box-playback  # View the output
   ```

The orchestrator will route requests to your local runner at `http://localhost:8000` instead of spinning up a Docker container.

---

## Best Practices

### Frame Tensor Format

- **Input**: `(B, H, W, C)` shape, `float32`, range `[-1, 1]`
- **Output**: Same format as input
- Convert inside your pipeline if your model expects different formats

### Async Operations

- Both `async def` and `def` work — the `@pipeline` decorator automatically runs sync methods in a thread pool so they won't block the event loop

### Error Handling

- Let exceptions propagate - the caller handles recovery
- The `ProcessGuardian` will restart stuck pipelines automatically

### Parameter Updates

- Return nothing from `on_update()` for instant updates
- For slow reloads, the runtime shows a loading overlay while the update is running

---

## Troubleshooting

### Common Issues

1. **"No container image found"**: Ensure your pipeline is registered in `docker.go`'s `livePipelineToImage` map.

2. **Models not found at runtime**: Check that `HF_HUB_OFFLINE=1` is set and models were prepared on the right path (`/models` when running in docker).

3. **CUDA out of memory**: The pipeline runs in an isolated subprocess - OOM errors will trigger a restart.

---

## Reference Implementation

For a complete working example, see [scope-runner](https://github.com/livepeer/scope-runner).

---

## Alternative: Batch Pipelines

This guide focuses on **live (streaming) pipelines** that process video frames in real-time. However, you can also implement **batch pipelines** for request/response workloads (e.g., text-to-image, audio-to-text).

Batch pipelines extend a different base class:

```python
from runner.pipelines.base import Pipeline

class MyBatchPipeline(Pipeline):
    name: str = "my-batch-pipeline"

    def __init__(self, model_id: str, model_dir: str = "/models"):
        self.model_id = model_id
        self.model_dir = model_dir
        # Load your model here

    def __call__(self, **kwargs) -> Any:
        """Process a single request and return the result."""
        # Your inference logic here
        return result

    @property
    def router(self) -> "APIRouter":
        """Return a FastAPI router with your endpoints."""
        from fastapi import APIRouter
        router = APIRouter()

        @router.post("/my-endpoint")
        async def my_endpoint(request: MyRequest) -> MyResponse:
            return self(**request.dict())

        return router
```

**Key differences from live pipelines:**

| Aspect | Live Pipeline | Batch Pipeline |
|--------|---------------|----------------|
| Base class | `runner.live.pipelines.Pipeline` | `runner.pipelines.base.Pipeline` |
| Processing | Continuous frame stream | Single request/response |
| Entry point | `start_app(pipeline_spec)` | `start_app(pipeline_instance)` |
| Model prep | `prepare_models()` classmethod | Manual in `dl_checkpoints.sh` |

> **⚠️ Note**: Batch pipelines do **not** have automatic model preparation support via `PREPARE_MODELS=1`. You must handle model downloading entirely through `dl_checkpoints.sh` using direct `hf download` commands or custom scripts. There is no `prepare_models()` hook for batch pipelines.

For batch pipeline examples, see the existing implementations in `runner/src/runner/pipelines/` (e.g., `text_to_image.py`, `audio_to_text.py`). There is no current example of such pipeline implemented on a separate project though.
