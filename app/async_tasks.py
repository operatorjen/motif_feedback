from __future__ import annotations

import asyncio
from contextlib import suppress


async def cancel_and_wait(task: asyncio.Task) -> None:
    """Cancel unfinished request work and always observe task completion."""
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError):
        await task
