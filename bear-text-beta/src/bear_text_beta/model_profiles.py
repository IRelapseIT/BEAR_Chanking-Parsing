"""Explicit retrieval templates; actual revisions are pinned by prepare-model."""

from .models import EmbeddingProfile


def model_profile(name: str) -> EmbeddingProfile:
    if name == "e5-small":
        return EmbeddingProfile(model_id="intfloat/multilingual-e5-small", vector_dimension=384,
                                query_prefix="query: ", document_prefix="passage: ",
                                similarity_prefix="query: ")
    if name == "bge-m3":
        # BGE-M3 supports 8192 tokens; 512 is our conservative operating budget.
        # Unlike BGE v1.5, M3 does not require a query instruction.
        return EmbeddingProfile(model_id="BAAI/bge-m3", vector_dimension=1024,
                                max_input_tokens=512, query_prefix="", document_prefix="",
                                similarity_prefix="")
    raise ValueError(f"Unsupported model template: {name}")
