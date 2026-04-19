from typing import Any, Dict


class WorldState:
    """
    Minimal world state container (Γ).
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}

    # ---------------------------------------------------------
    # BASIC ACCESS
    # ---------------------------------------------------------
    def get(self, key: str, default=None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        self.data.update(updates)

    # ---------------------------------------------------------
    # UTIL
    # ---------------------------------------------------------
    def __repr__(self) -> str:
        return f"WorldState({self.data})"