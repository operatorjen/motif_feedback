from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field


@dataclass
class CodeRunSession:
    project_id: str
    path: str
    events: list[dict] = field(default_factory=list)
    input_bytes: int = 0
    events_lock: threading.Lock = field(default_factory=threading.Lock)
    input_queue: queue.Queue[dict] = field(default_factory=queue.Queue)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def record_output(self, event_type: str, text: str) -> None:
        with self.events_lock:
            self.events.append({"type": event_type, "text": text})

    def snapshot(self, after: int = 0) -> tuple[list[dict], int]:
        with self.events_lock:
            return list(self.events[after:]), len(self.events)

    def record_input(self, text: str, size: int, maximum: int) -> bool:
        with self.events_lock:
            if self.input_bytes + size > maximum:
                return False
            self.input_bytes += size
            self.events.append({"type": "stdin", "text": text})
        return True

    def send(self, command: dict) -> None:
        self.input_queue.put(command)

    def cancel(self) -> None:
        self.cancel_event.set()
        self.send({"action": "cancel"})
