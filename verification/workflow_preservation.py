"""Compare read-only primary sources and inherited model work to saved snapshots."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
snapshots = json.loads((ROOT / "verification/model-comparison/source-before.json").read_text(encoding="utf-8"))
primary = ROOT.parent / "BEAR_Chanking-Parsing"
expected_primary = snapshots[str(primary)]
changed_primary = [name for name, digest in expected_primary.items()
                   if hashlib.sha256((primary / name).read_bytes()).hexdigest() != digest]
inherited = json.loads((ROOT / "verification/model-comparison/verification-summary.json").read_text(encoding="utf-8"))
expected_copy = inherited["bge_source_sha256"]
allowed = {"api.py", "runtime.py", "indexing.py", "storage.py", "process_worker.py"}
unexpected = [name for name, digest in expected_copy.items()
              if Path(name).name not in allowed and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
report = {"primary_changed_files": changed_primary, "unexpected_changes_to_inherited_model_work": unexpected,
          "primary_files_checked": len(expected_primary), "inherited_files_checked": len(expected_copy),
          "intentional_existing_source_changes": sorted(allowed),
          "weights_verification_reused": "verification/model-comparison/weights-verified.json",
          "comparison_reused": "docs/MODEL_COMPARISON_2026-11-06.md"}
(ROOT / "verification/workflow/preservation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
assert not changed_primary and not unexpected, report
print("PRIMARY_AND_MODEL_WORK_PRESERVED")
