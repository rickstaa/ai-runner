"""Dev server for testing live pipelines without go-livepeer.

Spins up an embedded trickle server, starts the pipeline subprocess,
and serves a browser-based test UI at /dev.

Usage:
    # Using a @pipeline-decorated module:
    python -m runner.dev --pipeline examples.live_video_to_video.green_shift:GreenShiftPipeline

    # Using a built-in pipeline name:
    python -m runner.dev --pipeline comfyui

    # Custom port:
    python -m runner.dev --pipeline examples.live_video_to_video.green_shift:GreenShiftPipeline --port 8000
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from runner.live.trickle.server import TrickleServer, create_trickle_router
from runner.live.pipelines import PipelineSpec, builtin_pipeline_spec
from runner.live.pipelines.loader import load_pipeline_class
from runner.live.log import config_logging

logger = logging.getLogger(__name__)

DEV_HTML_PATH = Path(__file__).parent / "static" / "dev.html"


def resolve_pipeline_spec(pipeline_arg: str) -> PipelineSpec:
    """Resolve a pipeline argument to a PipelineSpec.

    Accepts:
      - A built-in pipeline name (e.g. "comfyui")
      - A module:class path (e.g. "examples.live_video_to_video.green_shift:GreenShiftPipeline")
    """
    # Try built-in name first
    spec = builtin_pipeline_spec(pipeline_arg)
    if spec:
        return spec

    # Try module:class path — load the class and check for _spec
    pipeline_cls = load_pipeline_class(pipeline_arg)
    if hasattr(pipeline_cls, "_spec"):
        return pipeline_cls._spec

    # Fall back: create a spec from the import path
    return PipelineSpec(
        name=pipeline_arg.split(":")[-1] if ":" in pipeline_arg else pipeline_arg,
        pipeline_cls=pipeline_arg,
    )


def create_dev_app(pipeline_spec: PipelineSpec, port: int = 8000) -> FastAPI:
    """Create a FastAPI app with embedded trickle server and dev UI."""
    trickle_server = TrickleServer()
    infer_process = None
    log_thread = None

    base_url = f"http://localhost:{port}"
    subscribe_url = f"{base_url}/trickle/input"
    publish_url = f"{base_url}/trickle/output"
    control_url = f"{base_url}/trickle/control"
    events_url = f"{base_url}/trickle/events"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal infer_process, log_thread

        # Start the infer subprocess pointing at our local trickle server
        cmd = [
            sys.executable, "-u", "-m", "runner.live.infer",
            "--pipeline", pipeline_spec.model_dump_json(),
            "--http-port", "8888",
            "--stream-protocol", "trickle",
            "--subscribe-url", subscribe_url,
            "--publish-url", publish_url,
            "--control-url", control_url,
            "--events-url", events_url,
        ]

        env = os.environ.copy()
        model_dir = os.getenv("MODEL_DIR", "")
        if model_dir:
            env.setdefault("HUGGINGFACE_HUB_CACHE", model_dir)
            env.setdefault("DIFFUSERS_CACHE", model_dir)

        logger.info(f"Starting infer subprocess for pipeline: {pipeline_spec.name}")
        logger.info(f"  Subscribe URL: {subscribe_url}")
        logger.info(f"  Publish URL:   {publish_url}")
        logger.info(f"  Control URL:   {control_url}")

        infer_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )

        def stream_logs():
            try:
                for line in infer_process.stdout:
                    sys.stderr.write(f"[infer] {line}")
            except Exception:
                pass

        log_thread = threading.Thread(target=stream_logs, daemon=True)
        log_thread.start()

        logger.info(f"Dev server ready at {base_url}/dev")
        yield

        # Shutdown
        if infer_process and infer_process.poll() is None:
            logger.info("Stopping infer subprocess...")
            infer_process.terminate()
            try:
                infer_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                infer_process.kill()

    app = FastAPI(lifespan=lifespan)

    # Mount trickle server
    trickle_router = create_trickle_router(trickle_server)
    app.include_router(trickle_router)

    # Dev UI
    @app.get("/dev", response_class=HTMLResponse)
    async def dev_page():
        if DEV_HTML_PATH.exists():
            return HTMLResponse(DEV_HTML_PATH.read_text())
        return HTMLResponse("<h1>dev.html not found</h1>", status_code=500)

    # Pipeline info endpoint for the dev UI
    @app.get("/dev/info")
    async def dev_info():
        return {
            "pipeline": pipeline_spec.name,
            "subscribe_url": subscribe_url,
            "publish_url": publish_url,
            "control_url": control_url,
            "events_url": events_url,
        }

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Dev server for testing live pipelines without go-livepeer"
    )
    parser.add_argument(
        "--pipeline", type=str, required=True,
        help="Pipeline name or module:class path (e.g. 'comfyui' or 'examples.live_video_to_video.green_shift:GreenShiftPipeline')",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config_logging(
        log_level=logging.DEBUG if args.verbose else logging.INFO,
    )

    pipeline_spec = resolve_pipeline_spec(args.pipeline)
    logger.info(f"Pipeline: {pipeline_spec.name} ({pipeline_spec.pipeline_cls})")

    import uvicorn
    app = create_dev_app(pipeline_spec, port=args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
