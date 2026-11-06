"""Opt-in local dense inference. Prefixes belong to the BEAR HTTP client, not this server."""
import argparse
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bear_text_beta.embeddings import load_profile

FAMILIES = {"BAAI/bge-m3": ("xlm-roberta", "cls"),
            "Qwen/Qwen3-Embedding-0.6B": ("qwen3", "last")}


def pool_hidden(hidden, mask, pooling):
    if pooling == "cls":
        return hidden[:, 0]
    if pooling != "last":
        raise ValueError("Unsupported pooling")
    import torch
    # Works for left and right padding; padded states must never become vectors.
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(mask == 0, -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError("Empty attention mask")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class DenseLocalEncoder:
    """Load pinned local weights only; no truncation and no model-specific prompt injection."""

    def __init__(self, model_dir, profile, *, max_tokens=480, threads=4, microbatch=4):
        if profile.model_id not in FAMILIES:
            raise ValueError("Local inference supports BGE-M3 and Qwen3-Embedding-0.6B")
        if not profile.model_revision or not profile.tokenizer_revision:
            raise ValueError("Model and tokenizer revisions must be pinned")
        if not 1 <= max_tokens <= profile.max_input_tokens or not 1 <= threads <= 32 or not 1 <= microbatch <= 16:
            raise ValueError("Invalid inference limits")
        import torch
        from transformers import AutoModel, AutoTokenizer
        if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
            raise ValueError("Use torch >= 2.6 for weights_only loading of local .bin files")
        torch.set_num_threads(threads)
        self.profile, self.max_tokens, self.microbatch = profile, max_tokens, microbatch
        self.pooling = FAMILIES[profile.model_id][1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = "left" if self.pooling == "last" else "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model, info = AutoModel.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, weights_only=True,
            dtype=torch.float32, low_cpu_mem_usage=True, output_loading_info=True,
        )
        if self.model.config.model_type != FAMILIES[profile.model_id][0]:
            raise ValueError("Weights architecture differs from the declared model")
        if self.model.config.hidden_size != profile.vector_dimension:
            raise ValueError("Weights dimension differs from the declared profile")
        missing = [key for key in info["missing_keys"] if not key.startswith("pooler.")]
        if missing or info.get("mismatched_keys"):
            raise ValueError("Incomplete model backbone weights")
        self.model.eval()
        self.lock = threading.Lock()
        self.metrics = {"requests": 0, "texts": 0, "tokens": 0}

    def encode(self, texts):
        if not isinstance(texts, list) or not 1 <= len(texts) <= 16:
            raise ValueError("Batch size must be 1..16")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        counts = [len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in texts]
        if any(count > self.max_tokens for count in counts):
            raise ValueError("Input exceeds token budget; truncation is disabled")
        import torch
        vectors = []
        with self.lock, torch.inference_mode():
            for start in range(0, len(texts), self.microbatch):
                inputs = self.tokenizer(texts[start:start + self.microbatch], padding=True,
                                        truncation=False, return_tensors="pt")
                options = {"use_cache": False} if self.pooling == "last" else {}
                hidden = self.model(**inputs, **options).last_hidden_state
                pooled = pool_hidden(hidden, inputs["attention_mask"], self.pooling).float()
                norms = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
                if not torch.isfinite(pooled).all() or not (norms > 0).all():
                    raise RuntimeError("Model returned invalid vectors")
                normalized = pooled / norms
                if normalized.shape[1] != self.profile.vector_dimension:
                    raise RuntimeError("Model returned wrong dimension")
                vectors.extend(normalized.tolist())
            self.metrics["requests"] += 1
            self.metrics["texts"] += len(texts)
            self.metrics["tokens"] += sum(counts)
        return vectors


def make_server(encoder, host="127.0.0.1", port=7998):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("This development inference server only binds to loopback")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, status=200):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") not in {"/v1/models", "/models", "/health"}:
                return self.reply({"error": "Unknown endpoint"}, 404)
            self.reply({"data": [{"id": encoder.profile.model_id,
                                  "revision": encoder.profile.model_revision,
                                  "dimension": encoder.profile.vector_dimension,
                                  "pooling": encoder.pooling}], "metrics": encoder.metrics,
                        "max_input_tokens": encoder.max_tokens})

        def do_POST(self):
            if self.path.rstrip("/") not in {"/v1/embeddings", "/embeddings"}:
                return self.reply({"error": "Unknown endpoint"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024**2:
                    raise ValueError("Request body must be 1 byte..1MiB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or body.get("model") != encoder.profile.model_id:
                    raise ValueError("Requested model differs from loaded weights")
                vectors = encoder.encode(body["input"])
                self.reply({"model": encoder.profile.model_id, "data": [
                    {"index": index, "embedding": vector} for index, vector in enumerate(vectors)]})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply({"error": str(exc)}, 422)
            except Exception:
                logging.exception("Local inference failed")
                self.reply({"error": "Local inference failed"}, 500)

        def log_message(self, *_):
            pass

    class LocalHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = False

        def server_bind(self):
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

    return LocalHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local BEAR dense embeddings (CPU float32, no downloads)")
    parser.add_argument("--model-dir", type=Path, default=Path("models/bge-m3"))
    parser.add_argument("--port", type=int, default=7998)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=480)
    args = parser.parse_args(argv)
    profile = load_profile(args.model_dir)
    encoder = DenseLocalEncoder(args.model_dir, profile, threads=args.threads,
                                microbatch=args.microbatch, max_tokens=args.max_tokens)
    server = make_server(encoder, port=args.port)
    print(f"BEAR local inference: http://127.0.0.1:{args.port}/v1 model={profile.model_id} pooling={encoder.pooling}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
