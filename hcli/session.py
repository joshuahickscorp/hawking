from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .persist import atomic_write_json


class Session:
    def __init__(self, session_id: Optional[str] = None, goal: str = "", runtime_count: int = 1, model: Optional[str] = None):
        self.id = session_id or str(uuid.uuid4())
        self.goal = goal
        self.runtime_count = runtime_count
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        self.steering: List[str] = []
        self.mission_id: Optional[str] = None
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "runtime_count": self.runtime_count,
            "model": self.model,
            "messages": self.messages,
            "steering": self.steering,
            "mission_id": self.mission_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        s = cls(session_id=data.get("id"), goal=data.get("goal", ""),
                runtime_count=data.get("runtime_count", 1), model=data.get("model"))
        s.messages = data.get("messages", [])
        s.steering = data.get("steering", [])
        s.mission_id = data.get("mission_id")
        s.created_at = data.get("created_at", s.created_at)
        return s


class SessionStore:
    def __init__(self, workspace: str):
        self.dir = os.path.join(workspace, ".hcli", "sessions")
        os.makedirs(self.dir, exist_ok=True)

    def save(self, session: Session):
        path = os.path.join(self.dir, f"{session.id}.json")
        atomic_write_json(path, session.to_dict())

    def load(self, session_id: str) -> Optional[Session]:
        path = os.path.join(self.dir, f"{session_id}.json")
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return Session.from_dict(json.load(f))
