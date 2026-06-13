from pathlib import Path


class CnnModelStatus:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    @property
    def exists(self) -> bool:
        return self.model_path.exists()

    @property
    def size_bytes(self) -> int:
        if not self.exists:
            return 0
        return self.model_path.stat().st_size

    @property
    def is_loadable_placeholder(self) -> bool:
        return self.exists and self.size_bytes > 0
