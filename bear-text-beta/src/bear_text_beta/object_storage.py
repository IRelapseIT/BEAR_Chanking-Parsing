"""Local beta adapter for the team's ObjectStorage port."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from .errors import StorageError


class LocalObjectStorage:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, object_key: str) -> Path:
        relative = Path(object_key)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise StorageError("Некорректный object_key")
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise StorageError("object_key выходит за пределы хранилища")
        return target

    def put_stream(self, object_key: str, source: BinaryIO) -> None:
        target = self._path(object_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".upload-",
                                         delete=False) as temp:
            temporary = Path(temp.name)
        try:
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def get_stream(self, object_key: str) -> BinaryIO:
        return self._path(object_key).open("rb")

    def delete(self, object_key: str) -> None:
        self._path(object_key).unlink(missing_ok=True)
