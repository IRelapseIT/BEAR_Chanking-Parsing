"""Real HTTP acceptance corpus; every case remains in the denominator."""
import concurrent.futures
import hashlib
import io
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import httpx
import psutil
from bear_text_beta.models import DocumentContext
from docx import Document
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification/workflow"
API = "http://127.0.0.1:8002"


def pdf(text=None, scan=False):
    stream = io.BytesIO()
    writer = canvas.Canvas(stream)
    if scan:
        from reportlab.lib.utils import ImageReader
        image = Image.new("RGB", (1600, 500), "white")
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)
        ImageDraw.Draw(image).text((80, 120), "SCAN POLICY: Refund within 27 days.", font=font, fill="black")
        writer.drawImage(ImageReader(image), 30, 400, width=540, height=170)
    else:
        writer.drawString(40, 700, text)
    writer.save()
    return stream.getvalue()


def docx(embedded=False):
    doc = Document()
    doc.add_heading("Delivery conditions", 1)
    doc.add_paragraph("Nord delivers parcels within 7 working days.", style="List Bullet")
    doc.sections[0].header.paragraphs[0].text = "Delivery company"
    doc.sections[0].footer.paragraphs[0].text = "Terms of delivery"
    table = doc.add_table(rows=3, cols=2)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "Delivery regions"
    for cell, value in [(table.cell(1, 0), "Nord"), (table.cell(1, 1), "7 working days"),
                        (table.cell(2, 0), "South"), (table.cell(2, 1), "18 working days")]:
        cell.text = value
    if embedded:
        image = io.BytesIO()
        Image.new("RGB", (50, 50), "red").save(image, format="PNG")
        image.seek(0)
        doc.add_picture(image)
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


def wait_job(client, job_id):
    for _ in range(1200):
        state = client.get(f"{API}/v1/jobs/{job_id}").raise_for_status().json()
        if state["worker_status"] in {"complete", "failed"}:
            return state
        time.sleep(.2)
    raise RuntimeError(f"Job timeout: {job_id}")


def main():
    OUT.mkdir(exist_ok=True)
    run = uuid4().hex[:10]
    cases = [
        ("refund.txt", "Магазин Норд возвращает деньги за 12 рабочих дней.".encode(), "За сколько дней Норд вернет деньги?", "12", "controlled"),
        ("backup.txt", "Архивные резервные копии хранятся 43 дня.".encode(), "Срок хранения архивных резервных копий?", "43", "controlled"),
        ("access.txt", "Пропуск в лабораторию оформляется через деканат.".encode(), "Где оформить пропуск в лабораторию?", "деканат", "controlled"),
        ("ssl.md", b"# NGINX SSL\n\nUse ssl_certificate to configure HTTPS.\n\n```nginx\nssl_certificate /etc/ssl/bear.pem;\n```", "Which file configures NGINX certificate?", "/etc/ssl/bear.pem", "controlled"),
        ("table.md", b"# Revenue\n\n| Year | Profit |\n|---|---|\n|2025|150 million RUB|\n|2026|180 million RUB|", "Profit for 2025?", "150", "controlled"),
        ("manual.html", b"<h1>Support</h1><p>Support uses helpdesk@example.test.</p>", "Support email address?", "helpdesk@example.test", "controlled"),
        ("table.htm", b"<h1>Hours</h1><table><tr><th>Office</th><th>Hours</th></tr><tr><td>East</td><td>09:30</td></tr></table>", "East office opening hours?", "09:30", "controlled"),
        ("delivery.docx", docx(), "Delivery time for Nord parcels?", "7", "controlled"),
        ("embedded.docx", docx(True), "Delivery conditions?", "7", "controlled_unsupported_object"),
        ("text.pdf", pdf("Annual leave lasts 28 calendar days."), "How long is annual leave?", "28", "controlled"),
        ("scan.pdf", pdf(scan=True), "Refund time in the scanned policy?", "27", "controlled_scan"),
    ]
    public = OUT / "corpus"
    for name, query, answer in [("dummy.pdf", "Dummy PDF file", "Dummy"),
                                 ("typescript.md", "What is TypeScript?", "TypeScript")]:
        cases.append((name, (public / name).read_bytes(), query, answer, "public_real_document"))
    cases.append(("broken.pdf", b"not-a-pdf", "bad", "bad", "invalid_control"))
    # Oracle and denominator are fixed before uploading any file.
    manifest = [{"file": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                 "query": query, "answer": answer, "kind": kind}
                for name, data, query, answer, kind in cases]
    (OUT / "corpus-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    results, search_times, queries = [], [], []
    with httpx.Client(timeout=60) as client:
        client.get(API + "/live").raise_for_status()
        for index, ((name, data, query, answer, kind), oracle) in enumerate(zip(cases, manifest, strict=True)):
            document_id = f"{run}-{index}"
            context = DocumentContext("workflow", run, document_id, 1, name)
            start = time.perf_counter()
            response = client.post(API + "/v1/documents", files={"file": (name, data)},
                                   data={"context": json.dumps(asdict(context))})
            response.raise_for_status()
            job_id = response.json()["job_id"]
            state = wait_job(client, job_id)
            elapsed = time.perf_counter() - start
            request = {"query": query, "tenant_id": "workflow", "space_id": run, "top_k": 5}
            start = time.perf_counter()
            hits = client.post(API + "/v1/search", json=request).raise_for_status().json()["results"]
            search_times.append(time.perf_counter() - start)
            rank = next((i + 1 for i, hit in enumerate(hits) if hit["document_id"] == document_id
                         and answer.casefold() in hit["text"].casefold()), 0)
            sources = any(hit["sources"] for hit in hits if hit["document_id"] == document_id)
            item = {**oracle, "job_id": job_id, "document_id": document_id, "state": state,
                    "seconds": elapsed, "rank_at_5": rank, "has_sources": sources,
                    "automatic_success": state["status"] == "searchable" and rank > 0 and sources}
            results.append(item)
            queries.append(request)
            print(name, state["status"], "rank", rank, flush=True)
        duplicate = client.post(API + "/v1/documents", files={"file": (cases[0][0], cases[0][1])},
                                data={"context": json.dumps(asdict(DocumentContext(
                                    "workflow", run, results[0]["document_id"], 1, cases[0][0])))})
        assert duplicate.json()["job_id"] == results[0]["job_id"]
        assert client.post(API + "/v1/search", json={**queries[0], "tenant_id": "foreign"}).json()["results"] == []
        assert client.post(API + "/v1/search", json={**queries[0], "space_id": "foreign"}).json()["results"] == []
        # Parallel retrieval uses the API process, while parsing/indexing belongs to the worker.
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(lambda q: client.post(API + "/v1/search", json=q).status_code, queries[:4]))
        parallel_seconds = time.perf_counter() - start
        assert parallel == [200] * 4
        start = time.perf_counter()
        client.post(API + "/v1/search", json=queries[0]).raise_for_status()
        cached_seconds = time.perf_counter() - start
    registry = json.loads((ROOT / "runtime/processes.json").read_text())
    memory = {}
    for name, info in registry.items():
        try:
            p = psutil.Process(info["pid"])
            memory[name] = (p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))) / 1024**2
        except psutil.NoSuchProcess:
            pass
    report = {"run": run, "documents": len(results), "automatic_successes": sum(r["automatic_success"] for r in results),
              "success_fraction": sum(r["automatic_success"] for r in results) / len(results),
              "recall_at_5_all_cases": sum(r["rank_at_5"] > 0 for r in results) / len(results),
              "mrr_at_5_all_cases": sum(1 / r["rank_at_5"] if r["rank_at_5"] else 0 for r in results) / len(results),
              "cases": results, "search_seconds": search_times, "cached_search_seconds": cached_seconds,
              "parallel_4_seconds": parallel_seconds, "rss_mib_snapshot": memory,
              "hardware": {"platform": platform.platform(), "cpu": platform.processor(),
                           "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                           "ram_gib": psutil.virtual_memory().total / 1024**3},
              "index_bytes": sum(p.stat().st_size for p in (ROOT / "runtime/qdrant-storage").rglob("*") if p.is_file()),
              "scope": "Small explicit corpus, not an agreed representative company benchmark. Invalid/review cases kept in denominator."}
    (OUT / "api-corpus-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), str(ROOT / "scripts/services.py"), "stop"], check=True)
    subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), str(ROOT / "scripts/services.py"), "start"], check=True,
                   stdout=subprocess.DEVNULL)
    with httpx.Client(timeout=60) as client:
        for _ in range(100):
            try:
                if client.get(API + "/live").is_success:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(.2)
        hits = client.post(API + "/v1/search", json=queries[0]).raise_for_status().json()["results"]
        assert any(h["document_id"] == results[0]["document_id"] for h in hits)
    report["restart_persistence_verified"] = True
    (OUT / "api-corpus-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REAL_API_BGE_QDRANT_WORKFLOW_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
