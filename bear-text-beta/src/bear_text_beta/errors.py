class BearError(Exception):
    """An expected, diagnosable processing failure."""


class ParseError(BearError):
    pass


class ValidationError(BearError):
    pass


class EmbeddingError(BearError):
    pass


class StorageError(BearError):
    pass


class ServiceUnavailableError(BearError):
    """An external parser/chunker failure that may succeed on retry."""
