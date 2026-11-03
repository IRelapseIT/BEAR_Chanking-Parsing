from uuid import UUID

from pydantic import TypeAdapter

from .models import ProcessingResult


def result_adapter() -> TypeAdapter:
    return TypeAdapter(ProcessingResult)


def load_result(data: dict) -> ProcessingResult:
    result = result_adapter().validate_python(data)
    UUID(result.generation_id)
    if result.context != result.parsed_document.context:
        raise ValueError("ProcessingResult context differs from ParsedDocument")
    return result
