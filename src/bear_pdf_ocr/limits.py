from dataclasses import dataclass


@dataclass(frozen=True)
class FileLimits:
    max_file_bytes: int = 50 * 1024 * 1024
    max_attachments: int = 100
    max_pages: int = 500
    max_blocks: int = 100_000
    max_text_chars: int = 10_000_000

    def __post_init__(self) -> None:
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError("File limits must be positive")
