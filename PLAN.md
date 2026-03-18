# Livepeer AI Runner — Consolidation & Dev Experience Plan

## Overview

Six phases that build on each other. Each phase produces a working, testable artifact before moving on.

---

## Phase 1: Test @pipeline Decorators Using ComfyUI Box

**Goal**: Prove the `@pipeline` decorator works with real ComfyUI box pipelines, not just toy examples.

### 1.1 Create a ComfyUI box pipeline using @pipeline decorator
- Today `live/comfyui/pipeline/pipeline.py` directly subclasses the `Pipeline` ABC
- Wrap it with `@pipeline` to prove the decorator handles the ComfyUI lifecycle (async workflow prompts, frame queues, client connection)
- Create `examples/live-video-to-video/comfyui_box.py`:
  ```python
  @pipeline(name="comfyui-box", params=ComfyUIParams)
  class ComfyUIBox:
      def on_ready(self, **kw):
          self.client = ComfyStreamClient(...)
      def transform(self, frame, params):
          return self.client.queue_prompt(frame, params.prompt)
      def on_stop(self):
          self.client.close()
  ```

### 1.2 Test using dev server
- `python -m runner.dev --pipeline examples.live_video_to_video.comfyui_box:ComfyUIBox`
- Verify: webcam → browser test UI → processed frames → browser
- Verify: control messages (workflow JSON updates) work via `/trickle/control`

### 1.3 Fix decorator gaps found during testing
- Handle async generators if ComfyUI client returns them
- Handle frame queue sizing for ComfyUI's variable latency
- Any lifecycle issues (model reload, workflow hot-swap)

**Deliverable**: ComfyUI box working through `@pipeline` decorator + dev server.

---

## Phase 2: Extend @pipeline Decorator to Batch Pipelines

**Goal**: Unify batch and live pipelines under one decorator system, replacing the old `Pipeline` ABC + `__call__` + hardcoded switch statement.

### 2.1 Add `@pipeline(mode="batch")` support
- Extend `create.py` to handle batch (request/response) mode:
  ```python
  @pipeline(name="text-to-image", mode="batch", params=TextToImageParams)
  class TextToImage:
      def on_ready(self, model_id: str, **kw):
          self.ldm = AutoPipelineForText2Image.from_pretrained(model_id)
      def transform(self, **kwargs):
          return self.ldm(**kwargs)
  ```
- Batch mode generates a `__call__(**kwargs)` instead of `put_video_frame`/`get_processed_video_frame`
- Batch mode generates a `router` property with standard POST endpoint
- Batch mode generates proper `name` property

### 2.2 Add auto-registration to replace hardcoded switch
- Replace `load_pipeline()` match/case in `app.py` with a registry:
  ```python
  # app.py
  def load_pipeline(pipeline: str, model_id: str) -> Pipeline:
      registry = get_pipeline_registry()
      if pipeline in registry:
          return registry[pipeline](model_id)
      # fallback to PipelineSpec for live pipelines
      ...
  ```
- `@pipeline` decorator auto-registers into the registry
- Scan a configurable directory for decorated pipelines at startup

### 2.3 Create batch pipeline using @pipeline decorator (proof of concept)
- Convert ONE existing batch pipeline (e.g., `audio_to_text.py`) to use `@pipeline(mode="batch")`
- Verify it works identically to the old version
- Verify the route auto-generation matches the hand-written route

### 2.4 Handle ComfyUI box batch
- ComfyUI box in batch mode: receive a single image + workflow, return processed image
- `@pipeline(name="comfyui-box-batch", mode="batch", params=ComfyUIBatchParams)`
- Test via standard HTTP POST endpoint

**Deliverable**: `@pipeline` works for both `mode="live"` and `mode="batch"`. One batch pipeline converted as proof.

---

## Phase 3: Remove Old Batch Pipelines, Migrate to Pipelines Repo

**Goal**: Clean up ai-runner by removing old batch pipeline implementations and moving them to a dedicated pipelines repo.

### 3.1 Convert remaining batch pipelines to @pipeline decorator
- `text_to_image.py` → `@pipeline(name="text-to-image", mode="batch")`
- `image_to_image.py` → `@pipeline(name="image-to-image", mode="batch")`
- `image_to_video.py` → `@pipeline(name="image-to-video", mode="batch")`
- `image_to_text.py` → `@pipeline(name="image-to-text", mode="batch")`
- `audio_to_text.py` → `@pipeline(name="audio-to-text", mode="batch")`
- `text_to_speech.py` → `@pipeline(name="text-to-speech", mode="batch")`
- `segment_anything_2.py` → `@pipeline(name="segment-anything-2", mode="batch")`
- `upscale.py` → `@pipeline(name="upscale", mode="batch")`
- `llm.py` → `@pipeline(name="llm", mode="batch")`

### 3.2 Remove old infrastructure from ai-runner
- Delete `runner/src/runner/pipelines/base.py` (old ABC)
- Delete hand-written route files in `runner/src/runner/routes/` (auto-generated now)
- Remove hardcoded `load_pipeline()` switch in `app.py`
- Keep one example pipeline inline (e.g., `noop.py` for live, one simple batch)

### 3.3 Move pipeline implementations to pipelines repo
- Create `livepeer/ai-pipelines` repo (or similar)
- Each pipeline is a standalone package:
  ```
  ai-pipelines/
  ├── text-to-image/
  │   ├── pipeline.py       # @pipeline decorated
  │   ├── requirements.txt  # diffusers, torch, etc.
  │   └── Dockerfile
  ├── audio-to-text/
  │   ├── pipeline.py
  │   ├── requirements.txt
  │   └── Dockerfile
  └── ...
  ```
- ai-runner stays as the **runtime** (framework + trickle + dev tools)
- Pipelines repo has the **implementations** (model-specific code)

### 3.4 Update Docker workflows
- Update `.github/workflows/ai-runner-docker-batch-*.yaml` to build from pipelines repo
- ai-runner base image + pipeline code layered on top

**Deliverable**: ai-runner is a clean runtime. All pipeline implementations live in pipelines repo. Old batch infrastructure deleted.

---

## Phase 4: Create `livepeer-trickle` Package

**Goal**: Consolidate 5 duplicate trickle implementations into one reusable package.

### 4.1 Create livepeer-trickle repo
```
livepeer-trickle/
├── python/
│   ├── src/livepeer_trickle/
│   │   ├── __init__.py        # Public API
│   │   ├── subscriber.py      # From pytrickle (Feb 2026, newest)
│   │   ├── publisher.py       # From pytrickle (Feb 2026, newest)
│   │   ├── server.py          # From ai-runner server.py (FastAPI)
│   │   └── protocol.py        # Constants: headers, status codes
│   ├── tests/
│   │   ├── test_subscriber.py
│   │   ├── test_publisher.py
│   │   └── test_server.py
│   └── pyproject.toml         # deps: aiohttp, fastapi (optional)
│
├── go/
│   ├── publisher.go           # From go-livepeer/trickle/
│   ├── subscriber.go          # From go-livepeer/trickle/
│   ├── server.go              # From go-livepeer/trickle/
│   ├── local_publisher.go     # From go-livepeer/trickle/
│   ├── local_subscriber.go    # From go-livepeer/trickle/
│   ├── cmd/trickle-server/
│   │   └── main.go            # Standalone binary (NEW)
│   ├── go.mod
│   └── *_test.go
│
└── protocol.md                # THE protocol specification
```

### 4.2 Python package: extract from pytrickle
- Take `subscriber.py` and `publisher.py` from `livepeer/pytrickle` (newest, tested)
- Take `server.py` from ai-runner (simpler, FastAPI-based)
- Add `protocol.py` with shared constants:
  ```python
  HEADER_SEQ = "Lp-Trickle-Seq"
  HEADER_CLOSED = "Lp-Trickle-Closed"
  HEADER_LATEST = "Lp-Trickle-Latest"
  STATUS_REDIRECT = 470
  DEFAULT_MIME_TYPE = "video/mp2t"
  ```
- Extract shared base class from subscriber/publisher (session, lock, context manager)
- Publish to PyPI as `livepeer-trickle`
- Dependencies: `aiohttp` only. `fastapi` as optional extra for server.

### 4.3 Go package: extract from go-livepeer
- Copy `trickle_publisher.go`, `trickle_subscriber.go`, `trickle_server.go`, `local_*.go` from `go-livepeer/trickle/`
- Remove go-livepeer-specific imports (should already be stdlib-only)
- Create `cmd/trickle-server/main.go` — standalone binary:
  ```go
  func main() {
      port := flag.Int("port", 7935, "port")
      mux := http.NewServeMux()
      trickle.NewServer(trickle.TrickleServerConfig{Mux: mux})
      http.ListenAndServe(fmt.Sprintf(":%d", *port), mux)
  }
  ```
- Publish Go module as `github.com/livepeer/livepeer-trickle`

### 4.4 Replace in ai-runner
- `pip install livepeer-trickle`
- Delete `runner/src/runner/live/trickle/trickle_subscriber.py`
- Delete `runner/src/runner/live/trickle/trickle_publisher.py`
- Delete `runner/src/runner/live/trickle/server.py`
- Update imports:
  ```python
  # Before
  from runner.live.trickle import TrickleSubscriber, TricklePublisher
  # After
  from livepeer_trickle import TrickleSubscriber, TricklePublisher
  ```
- Keep in ai-runner: `media.py`, `encoder.py`, `decoder.py`, `frame.py` (domain-specific)

### 4.5 Replace in go-livepeer
- `go get github.com/livepeer/livepeer-trickle`
- Replace `go-livepeer/trickle/` imports with `livepeer-trickle/go`
- Delete vendored code

### 4.6 Replace in other consumers
- `livepeer/pytrickle` → depend on `livepeer-trickle`, remove vendored subscriber/publisher
- `j0sh/livepeer-python-gateway` → depend on `livepeer-trickle`
- `j0sh/http-trickle` → archive

**Deliverable**: `livepeer-trickle` published on PyPI and as Go module. All consumers use it. 5 implementations → 1.

---

## Phase 5: Create `livepeer-dev` Client

**Goal**: A standalone dev tool that lets you test any pipeline locally with FPS metrics, directly on the runner, before publishing.

### 5.1 Enhance dev server with metrics
- Add FPS tracking to dev server:
  - Input FPS (frames arriving from webcam/source)
  - Output FPS (frames leaving pipeline)
  - Pipeline latency (per-frame processing time)
  - Queue depth (backpressure indicator)
- Subscribe to events channel (`/trickle/events`) and display in dev UI
- Add metrics endpoint (`/dev/metrics`) with JSON stats

### 5.2 Improve dev UI
- Real-time FPS overlay on video output
- Latency graph (last 60 seconds)
- Parameter controls (auto-generated from `BaseParams` schema)
- Side-by-side input/output view
- Start/stop/restart pipeline controls

### 5.3 Add test sources beyond webcam
- File input: `--input video.mp4` → feed frames from file
- Test pattern: `--input test-pattern` → generated color bars
- RTMP input: `--input rtmp://...` → receive from OBS/FFmpeg
- Loop mode: `--loop` → repeat source file

### 5.4 Add benchmark mode
- `python -m runner.dev --pipeline my_pipeline --benchmark --input test-pattern --duration 30`
- Runs for N seconds, reports:
  - Average FPS
  - P50/P95/P99 frame latency
  - Memory usage (GPU + CPU)
  - Frame drops
- Outputs JSON report for CI integration

### 5.5 Sidecar mode with Go binary
- `python -m runner.dev --pipeline my_pipeline --trickle-server=go`
- Auto-detects `trickle-server` binary (from livepeer-trickle)
- Falls back to Python server if not found
- Logs which server is being used

### 5.6 Package as standalone CLI
- `pip install livepeer-dev` or part of `pip install ai-runner[dev]`
- Command: `livepeer-dev --pipeline my_module:MyPipeline`
- Works without cloning ai-runner repo

**Deliverable**: Full local dev experience with FPS metrics, benchmark mode, multiple inputs, sidecar support.

---

## Phase 6: End-to-End Validation

**Goal**: Verify everything works together.

### 6.1 Test matrix
- [ ] `@pipeline(mode="live")` + dev server + webcam → green_shift example
- [ ] `@pipeline(mode="live")` + dev server + ComfyUI box → comfyui_box example
- [ ] `@pipeline(mode="batch")` + HTTP POST → text-to-image
- [ ] `@pipeline(mode="batch")` + HTTP POST → ComfyUI box batch
- [ ] `livepeer-trickle` Python subscriber/publisher + Go server
- [ ] `livepeer-trickle` Python subscriber/publisher + Python server
- [ ] Dev server benchmark mode with metrics
- [ ] Dev server with Go trickle-server sidecar
- [ ] Pipeline from external pipelines repo loaded into runner

### 6.2 Documentation
- Update `docs/custom-pipeline.md` with new `@pipeline` decorator API
- Add `docs/dev-server.md` with dev tool usage
- Update `AGENTS.md` with new architecture diagram

### 6.3 Migration guide
- For existing batch pipeline authors: how to convert to `@pipeline`
- For pytrickle users: how to switch to `livepeer-trickle`
- For go-livepeer: how to import from `livepeer-trickle`

---

## Dependency Graph

```
Phase 1 (decorators + box)
    ↓
Phase 2 (extend to batch)
    ↓
Phase 3 (remove old, pipelines repo)

Phase 4 (livepeer-trickle) ← can run in parallel with Phases 1-3
    ↓
Phase 5 (livepeer-dev) ← depends on Phase 4 for sidecar
    ↓
Phase 6 (validation) ← depends on all
```

**Phases 1-3 and Phase 4 are independent and can be worked on in parallel.**

---

## Repos After Completion

| Repo | Purpose |
|------|---------|
| `livepeer/ai-runner` | Runtime framework: @pipeline decorator, dev server, trickle glue, FFmpeg |
| `livepeer/ai-pipelines` | Pipeline implementations: text-to-image, audio-to-text, etc. |
| `livepeer/livepeer-trickle` | Protocol package: subscriber, publisher, server (Python + Go) |
| `livepeer/go-livepeer` | Orchestrator/gateway: imports livepeer-trickle |
| `livepeer/pytrickle` | Archived or thin wrapper over livepeer-trickle |
| `j0sh/http-trickle` | Archived |
