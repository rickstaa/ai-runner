"""Embedded Trickle server for dev/testing without go-livepeer.

Implements the minimal Trickle HTTP protocol:
  POST   /{channel}/{seq}  - Publish a segment (chunked/streaming body)
  GET    /{channel}/{seq}   - Subscribe to a segment (blocks until available)
  DELETE /{channel}         - Close a channel

Headers:
  Lp-Trickle-Seq:    Sequence number of the segment
  Lp-Trickle-Closed: Present when the channel is closed
  Lp-Trickle-Latest: Returned with 470 to indicate the latest available index
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


class TrickleChannel:
    """In-memory store for a single trickle channel."""

    def __init__(self):
        self.segments: dict[int, bytes] = {}
        self.events: dict[int, asyncio.Event] = {}
        self.latest_seq: int = -1
        self.closed: bool = False

    def _get_event(self, seq: int) -> asyncio.Event:
        if seq not in self.events:
            self.events[seq] = asyncio.Event()
        return self.events[seq]

    def put(self, seq: int, data: bytes):
        self.segments[seq] = data
        if seq > self.latest_seq:
            self.latest_seq = seq
        self._get_event(seq).set()
        self._cleanup(seq)

    async def get(self, seq: int, timeout: float = 30.0) -> Optional[bytes]:
        event = self._get_event(seq)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return self.segments.get(seq)

    def close(self):
        self.closed = True
        for event in self.events.values():
            event.set()

    def _cleanup(self, current_seq: int):
        """Remove segments older than 10 behind current to prevent memory leak."""
        stale = [s for s in self.segments if s < current_seq - 10]
        for s in stale:
            del self.segments[s]
            self.events.pop(s, None)


class TrickleServer:
    """Manages multiple trickle channels."""

    def __init__(self):
        self.channels: dict[str, TrickleChannel] = {}

    def get_or_create(self, name: str) -> TrickleChannel:
        if name not in self.channels:
            self.channels[name] = TrickleChannel()
        return self.channels[name]

    def remove(self, name: str):
        ch = self.channels.pop(name, None)
        if ch:
            ch.close()


def create_trickle_router(server: TrickleServer, prefix: str = "/trickle") -> APIRouter:
    """Create FastAPI router with trickle protocol endpoints."""
    router = APIRouter(prefix=prefix)

    @router.post("/{channel}/{seq}")
    async def publish_segment(channel: str, seq: int, request: Request):
        ch = server.get_or_create(channel)
        body = await request.body()
        ch.put(seq, body)
        logger.debug(f"Trickle publish channel={channel} seq={seq} bytes={len(body)}")
        return Response(
            status_code=200,
            headers={"Lp-Trickle-Seq": str(seq)},
        )

    @router.get("/{channel}/{seq}")
    async def subscribe_segment(channel: str, seq: int):
        ch = server.get_or_create(channel)

        # Handle negative index (start from latest)
        if seq < 0:
            if ch.latest_seq >= 0:
                seq = ch.latest_seq
            else:
                seq = 0

        # If requesting an old segment, redirect to latest
        if ch.latest_seq >= 0 and seq < ch.latest_seq - 10:
            return Response(
                status_code=470,
                headers={"Lp-Trickle-Latest": str(ch.latest_seq)},
            )

        data = await ch.get(seq)

        if ch.closed:
            return Response(
                status_code=200,
                content=data or b"",
                media_type="video/mp2t",
                headers={
                    "Lp-Trickle-Seq": str(seq),
                    "Lp-Trickle-Closed": "true",
                },
            )

        if data is None:
            return Response(status_code=404)

        return Response(
            status_code=200,
            content=data,
            media_type="video/mp2t",
            headers={"Lp-Trickle-Seq": str(seq)},
        )

    @router.delete("/{channel}")
    async def close_channel(channel: str):
        server.remove(channel)
        logger.info(f"Trickle channel closed: {channel}")
        return Response(status_code=200)

    return router
