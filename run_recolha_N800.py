#!/usr/bin/env python3
"""
Recolha N=800 — protocolo LLM v1.6 FINAL (Stimuli v1.1).

Reutiliza lógica de run_piloto_llm.py + build_prompts_N800.py.
NÃO modifica CSV LOCKED (personas / randomização / base_experimental).
NÃO escreve em pilot_outputs/.
NÃO inventa resultados empíricos.
OPENAI_API_KEY só via ambiente; OpenAI(max_retries=0).

Uso:
  python3 run_recolha_N800.py --dry-run
  python3 run_recolha_N800.py --dry-run --render-all
  python3 run_recolha_N800.py --only P037
  python3 run_recolha_N800.py --limit 4 --from-id P001
  python3 run_recolha_N800.py --resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import py_compile
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import build_prompts_N800 as bp  # noqa: E402
import run_piloto_llm as piloto  # noqa: E402

BASE_N800 = ROOT / "base_experimental_N800_seed20260921801.csv"
STIMULI_PATH = ROOT / "stimuli_locked.json"
OUT_DIR = ROOT / "recolha_N800_outputs"
RAW_CSV = OUT_DIR / "raw_responses.csv"
VALIDATION_JSON = OUT_DIR / "recolha_validation_report.json"
DRY_RUN_SUMMARY = OUT_DIR / "dry_run_summary.json"
RAW_API_DIR = OUT_DIR / "raw_api"
RENDERED_DIR = OUT_DIR / "rendered_prompts"
ANALYSIS_SCAFFOLD = OUT_DIR / "analysis_dataset.csv"

LOCKED_MD5 = {
    "personas_v3_1_N800_seed20260921800.csv": "14c8839535451b47be14267697069182",
    "randomizacao_N800_seed20260921801.csv": "4814fd05c030987a1365efda9aeb323d",
    "base_experimental_N800_seed20260921801.csv": "c45e134c05023b946b31709e209d70e7",
}

CONDITION_MAP = piloto.CONDITION_MAP
FORBIDDEN_IN_PROMPTS = piloto.FORBIDDEN_IN_PROMPTS
CONDITION_CODE_RE = piloto.CONDITION_CODE_RE
TURN1_MC_FORBIDDEN = piloto.TURN1_MC_FORBIDDEN
TURN2_MUST_NOT_CONTAIN = piloto.TURN2_MUST_NOT_CONTAIN

N_EXPECTED = 800
PER_CELL = 200


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_n800_base(path: Path = BASE_N800) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def filter_rows(
    rows: list[dict[str, str]],
    *,
    only: str | None = None,
    limit: int | None = None,
    from_id: str | None = None,
    to_id: str | None = None,
) -> list[dict[str, str]]:
    out = list(rows)
    if only:
        key = only.strip().upper()
        out = [r for r in out if r["persona_id"].upper() == key]
        if not out:
            raise SystemExit(f"Erro: persona_id {key} não encontrado na base N=800.")
    if from_id:
        fid = from_id.strip().upper()
        out = [r for r in out if r["persona_id"].upper() >= fid]
    if to_id:
        tid = to_id.strip().upper()
        out = [r for r in out if r["persona_id"].upper() <= tid]
    if limit is not None:
        if limit < 0:
            raise SystemExit("Erro: --limit deve ser >= 0.")
        out = out[:limit]
    return out


def persona_complete(row: dict[str, str]) -> bool:
    """Completo se T1+T2 parse status preenchidos, ou technical_failure com T1 tentado."""
    tf = (row.get("technical_failure") or "").strip()
    t1 = (row.get("turn1_parse_status") or "").strip()
    t2 = (row.get("turn2_parse_status") or "").strip()
    if t1 and t2:
        return True
    # Tech fail after T1 attempt (may lack T2) — treat as complete for resume skip
    if tf and t1:
        return True
    # Tech fail on T1 before parse status set
    if tf and "turn1" in tf.lower():
        return True
    return False


def load_completed_ids(path: Path = RAW_CSV) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if persona_complete(r):
                done.add(r["persona_id"])
    return done


def write_raw_api(persona_id: str, turn_label: str, payload: Any) -> Path:
    RAW_API_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_API_DIR / f"{persona_id}_{turn_label}.json"
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    elif isinstance(payload, dict):
        data = payload
    else:
        data = {"repr": repr(payload)}
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


def upsert_raw_csv(rec: dict[str, Any], fieldnames: list[str]) -> None:
    """Append or update row for persona_id; flush to disk immediately."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if RAW_CSV.exists():
        with open(RAW_CSV, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))
    pid = str(rec["persona_id"])
    serialized = {k: "" if rec.get(k) is None else str(rec.get(k, "")) for k in fieldnames}
    replaced = False
    new_rows: list[dict[str, str]] = []
    for r in rows:
        if r.get("persona_id") == pid:
            new_rows.append(serialized)
            replaced = True
        else:
            new_rows.append({k: r.get(k, "") for k in fieldnames})
    if not replaced:
        new_rows.append(serialized)
    with open(RAW_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(new_rows)
        f.flush()
        os.fsync(f.fileno())


def run_persona_live(
    client: Any,
    stimuli: dict,
    row: dict[str, str],
    api_cfg: dict[str, Any],
    schemas: dict[str, Any],
    fieldnames: list[str],
) -> dict[str, Any]:
    """Same API shape as piloto; writes to recolha_N800_outputs/."""
    system = bp.build_system_prompt(stimuli)
    turn1 = bp.build_turn1_user(stimuli, row)
    turn2 = bp.build_turn2_user(stimuli, row)
    rec = piloto.empty_audit_row(stimuli, row, system, turn1, turn2, api_cfg)
    pid = row["persona_id"]
    model = api_cfg["modelo"]

    # --- Turn 1 ---
    try:
        resp1 = piloto.call_responses(
            client,
            model=model,
            instructions=system,
            input_text=turn1,
            schema_obj=schemas["turn1"],
            api_cfg=api_cfg,
            previous_response_id=None,
        )
    except Exception as exc:
        rec["technical_failure"] = f"turn1_api_error: {type(exc).__name__}: {exc}"
        rec["turn1_parse_status"] = "technical_failure"
        write_raw_api(
            pid, "turn1_error", {"error": str(exc), "traceback": traceback.format_exc()}
        )
        upsert_raw_csv(rec, fieldnames)
        return rec

    raw1 = piloto.extract_output_text(resp1)
    rid1 = str(piloto.resp_attr(resp1, "id", default=""))
    rec["turn1_raw"] = raw1
    rec["turn1_response_id"] = rid1
    rec["api_model_returned"] = str(piloto.resp_attr(resp1, "model", default=""))
    rec["service_tier"] = str(piloto.resp_attr(resp1, "service_tier", default=""))
    rec["turn1_finish_status"] = str(
        piloto.resp_attr(resp1, "status", "finish_reason", default="")
    )
    in_t, out_t = piloto.usage_tokens(resp1)
    rec["input_tokens_turn1"] = in_t
    rec["output_tokens_turn1"] = out_t
    write_raw_api(pid, "turn1", resp1)

    status1, parsed1 = piloto.parse_turn1(raw1)
    rec["turn1_retry_flag"] = "0"
    if status1 != "valid":
        retry_msg = bp.turn1_retry_message(stimuli)
        try:
            resp1r = piloto.call_responses(
                client,
                model=model,
                instructions=system,
                input_text=retry_msg,
                schema_obj=schemas["turn1"],
                api_cfg=api_cfg,
                previous_response_id=rid1 or None,
            )
            raw1r = piloto.extract_output_text(resp1r)
            rid1r = str(piloto.resp_attr(resp1r, "id", default=""))
            write_raw_api(pid, "turn1_retry", resp1r)
            rec["turn1_retry_flag"] = "1"
            rec["turn1_raw"] = json.dumps(
                {"original": raw1, "retry": raw1r}, ensure_ascii=False
            )
            status1, parsed1 = piloto.parse_turn1(raw1r)
            if rid1r:
                rid1 = rid1r
                rec["turn1_response_id"] = rid1
            in_t2, out_t2 = piloto.usage_tokens(resp1r)
            try:
                rec["input_tokens_turn1"] = int(in_t or 0) + int(in_t2 or 0)
                rec["output_tokens_turn1"] = int(out_t or 0) + int(out_t2 or 0)
            except (TypeError, ValueError):
                pass
            rec["turn1_finish_status"] = str(
                piloto.resp_attr(
                    resp1r, "status", "finish_reason", default=rec["turn1_finish_status"]
                )
            )
        except Exception as exc:
            rec["technical_failure"] = f"turn1_retry_api_error: {type(exc).__name__}: {exc}"
            rec["turn1_parse_status"] = "technical_failure"
            write_raw_api(
                pid,
                "turn1_retry_error",
                {"error": str(exc), "traceback": traceback.format_exc()},
            )
            upsert_raw_csv(rec, fieldnames)
            return rec

    rec["turn1_parse_status"] = status1
    if parsed1:
        for k in ("BI1", "BI2", "BI3", "BI4"):
            rec[k] = parsed1[k]
    # Persist IMMEDIATELY before turn 2
    upsert_raw_csv(rec, fieldnames)

    if rec.get("technical_failure"):
        return rec

    # --- Turn 2 (even if turn1 invalid, if not technical_failure) ---
    try:
        resp2 = piloto.call_responses(
            client,
            model=model,
            instructions=system,
            input_text=turn2,
            schema_obj=schemas["turn2"],
            api_cfg=api_cfg,
            previous_response_id=rid1 or None,
        )
    except Exception as exc:
        rec["technical_failure"] = f"turn2_api_error: {type(exc).__name__}: {exc}"
        rec["turn2_parse_status"] = "technical_failure"
        write_raw_api(
            pid, "turn2_error", {"error": str(exc), "traceback": traceback.format_exc()}
        )
        upsert_raw_csv(rec, fieldnames)
        return rec

    raw2 = piloto.extract_output_text(resp2)
    rid2 = str(piloto.resp_attr(resp2, "id", default=""))
    rec["turn2_raw"] = raw2
    rec["turn2_response_id"] = rid2
    if not rec.get("api_model_returned"):
        rec["api_model_returned"] = str(piloto.resp_attr(resp2, "model", default=""))
    if not rec.get("service_tier"):
        rec["service_tier"] = str(piloto.resp_attr(resp2, "service_tier", default=""))
    rec["turn2_finish_status"] = str(
        piloto.resp_attr(resp2, "status", "finish_reason", default="")
    )
    in2, out2 = piloto.usage_tokens(resp2)
    rec["input_tokens_turn2"] = in2
    rec["output_tokens_turn2"] = out2
    write_raw_api(pid, "turn2", resp2)

    status2, parsed2 = piloto.parse_turn2(raw2)
    rec["turn2_retry_flag"] = "0"
    if status2 != "valid":
        retry_msg = bp.turn2_retry_message(stimuli)
        try:
            resp2r = piloto.call_responses(
                client,
                model=model,
                instructions=system,
                input_text=retry_msg,
                schema_obj=schemas["turn2"],
                api_cfg=api_cfg,
                previous_response_id=rid2 or rid1 or None,
            )
            raw2r = piloto.extract_output_text(resp2r)
            rid2r = str(piloto.resp_attr(resp2r, "id", default=""))
            write_raw_api(pid, "turn2_retry", resp2r)
            rec["turn2_retry_flag"] = "1"
            rec["turn2_raw"] = json.dumps(
                {"original": raw2, "retry": raw2r}, ensure_ascii=False
            )
            status2, parsed2 = piloto.parse_turn2(raw2r)
            if rid2r:
                rec["turn2_response_id"] = rid2r
            in2r, out2r = piloto.usage_tokens(resp2r)
            try:
                rec["input_tokens_turn2"] = int(in2 or 0) + int(in2r or 0)
                rec["output_tokens_turn2"] = int(out2 or 0) + int(out2r or 0)
            except (TypeError, ValueError):
                pass
            rec["turn2_finish_status"] = str(
                piloto.resp_attr(
                    resp2r, "status", "finish_reason", default=rec["turn2_finish_status"]
                )
            )
        except Exception as exc:
            rec["technical_failure"] = f"turn2_retry_api_error: {type(exc).__name__}: {exc}"
            rec["turn2_parse_status"] = "technical_failure"
            write_raw_api(
                pid,
                "turn2_retry_error",
                {"error": str(exc), "traceback": traceback.format_exc()},
            )
            upsert_raw_csv(rec, fieldnames)
            return rec

    rec["turn2_parse_status"] = status2
    if parsed2:
        for k in ("MC_FRAME", "CE1", "CE2", "CE3"):
            rec[k] = parsed2[k]
    upsert_raw_csv(rec, fieldnames)
    return rec


def _prompt_checksum(system: str, turn1: str, turn2: str) -> str:
    blob = (system + "\n---\n" + turn1 + "\n---\n" + turn2).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def check_prompt_leaks(pid: str, system: str, t1: str, t2: str) -> list[str]:
    hits: list[str] = []
    for blob_name, blob in (("system", system), ("turn1", t1), ("turn2", t2)):
        for tok in FORBIDDEN_IN_PROMPTS:
            if tok in blob:
                hits.append(f"{pid}/{blob_name}: forbidden token {tok!r}")
        if CONDITION_CODE_RE.search(blob):
            hits.append(f"{pid}/{blob_name}: condition code C1–C4 leak")
    for tok in TURN1_MC_FORBIDDEN:
        if tok in t1:
            hits.append(f"{pid}/turn1: MC content leak {tok!r}")
    for tok in TURN2_MUST_NOT_CONTAIN:
        if tok in t2:
            hits.append(f"{pid}/turn2: must-not-contain {tok!r}")
    if "Solicitar este crédito à habitação" in t2:
        hits.append(f"{pid}/turn2: BI wording re-sent")
    return hits


def run_dry_run(
    *,
    only: str | None = None,
    limit: int | None = None,
    from_id: str | None = None,
    to_id: str | None = None,
    render_all: bool = False,
) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    for script in (
        ROOT / "build_prompts_N800.py",
        ROOT / "run_piloto_llm.py",
        ROOT / "run_recolha_N800.py",
    ):
        try:
            py_compile.compile(str(script), doraise=True)
            checks[f"py_compile:{script.name}"] = "ok"
        except py_compile.PyCompileError as e:
            errors.append(f"py_compile failed for {script.name}: {e}")
            checks[f"py_compile:{script.name}"] = "fail"

    if not BASE_N800.exists():
        errors.append(f"Missing {BASE_N800}")
    if not STIMULI_PATH.exists():
        errors.append(f"Missing {STIMULI_PATH}")

    # Locked MD5s
    md5_ok: dict[str, str] = {}
    for name, expected in LOCKED_MD5.items():
        p = ROOT / name
        if not p.exists():
            errors.append(f"Locked file missing: {name}")
            continue
        got = file_md5(p)
        md5_ok[name] = got
        if got != expected:
            errors.append(f"MD5 mismatch {name}: got {got} expected {expected}")
    checks["locked_md5"] = md5_ok

    if errors:
        _write_dry_validation(False, checks, errors, warnings)
        print("DRY-RUN FAIL", file=sys.stderr)
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1

    stimuli = bp.load_stimuli()
    rows_all = load_n800_base()
    rows = filter_rows(
        rows_all, only=only, limit=limit, from_id=from_id, to_id=to_id
    )

    checks["N"] = len(rows_all)
    if len(rows_all) != N_EXPECTED:
        errors.append(f"Expected N={N_EXPECTED}, got {len(rows_all)}")

    ids = [r["persona_id"] for r in rows_all]
    expected_ids = [f"P{i:03d}" for i in range(1, N_EXPECTED + 1)]
    if ids != expected_ids:
        # allow if sorted order differs but set matches
        if sorted(ids) != expected_ids:
            errors.append(
                f"persona_ids must be P001–P800 uniquely; "
                f"got {len(ids)} ids, first={ids[:3]}, last={ids[-3:]}"
            )
        elif ids != expected_ids:
            warnings.append("persona_ids not in P001…P800 sequential file order")

    for pid in ids:
        if pid.upper().startswith("TEST"):
            errors.append(f"Forbidden TEST* id in N=800 base: {pid}")
        if not re.match(r"^P\d{3}$", pid):
            errors.append(f"ID must match Pxxx: {pid}")

    cond_counts = Counter(r["condicao"] for r in rows_all)
    checks["condicao_counts"] = dict(sorted(cond_counts.items()))
    for c in ("C1", "C2", "C3", "C4"):
        if cond_counts.get(c, 0) != PER_CELL:
            errors.append(
                f"Expected {PER_CELL} personas in {c}, got {cond_counts.get(c, 0)}"
            )

    for r in rows_all:
        pid = r["persona_id"]
        c = r["condicao"]
        if c not in CONDITION_MAP:
            errors.append(f"{pid}: unknown condicao {c!r}")
            continue
        exp_ciclo, exp_val = CONDITION_MAP[c]
        if r["ciclo_id"] != exp_ciclo or r["valencia_id"] != exp_val:
            errors.append(
                f"{pid}: ciclo_id/valencia_id={r['ciclo_id']}/{r['valencia_id']} "
                f"expected {exp_ciclo}/{exp_val} for {c}"
            )

    proto = stimuli.get("protocolo_versao", "")
    checks["protocolo_versao"] = proto
    if "1.6" not in str(proto):
        errors.append(f"protocolo_versao missing 1.6: {proto!r}")
    stim_v = stimuli.get("stimuli_versao", "")
    checks["stimuli_versao"] = stim_v
    if str(stim_v) != "1.1":
        warnings.append(f"stimuli_versao expected 1.1, got {stim_v!r}")

    api_cfg = bp.get_api_config(stimuli)
    checks["api_config_keys"] = sorted(api_cfg.keys())
    checks["model"] = api_cfg.get("modelo")
    if not api_cfg:
        errors.append("api_config missing/empty")
    for req in ("modelo", "temperature", "reasoning_effort", "max_output_tokens", "store"):
        if req not in api_cfg:
            errors.append(f"api_config missing key: {req}")

    try:
        computed = bp.schema_hash(stimuli)
        stored = bp.get_schemas(stimuli).get("schema_hash_sha256")
        checks["schema_hash"] = computed
        checks["schema_hash_stored"] = stored
        if stored and stored != computed:
            errors.append(f"schema_hash mismatch stored={stored} computed={computed}")
    except Exception as e:
        errors.append(f"schema_hash error: {e}")

    # Build all prompts (or filtered subset for --only/--limit); leak-check all built
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    system = bp.build_system_prompt(stimuli)
    build_rows = rows if (only or limit is not None or from_id or to_id) else rows_all
    # Prefer validating ALL 800 when no filter — required for full dry-run
    if not (only or limit is not None or from_id or to_id):
        build_rows = rows_all

    checksums: dict[str, str] = {}
    first_by_cond: dict[str, dict[str, str]] = {}
    leak_hits: list[str] = []
    n_built = 0

    for r in build_rows:
        pid = r["persona_id"]
        try:
            t1 = bp.build_turn1_user(stimuli, r)
            t2 = bp.build_turn2_user(stimuli, r)
        except Exception as e:
            errors.append(f"prompt build failed for {pid}: {e}")
            continue
        n_built += 1
        checksums[pid] = _prompt_checksum(system, t1, t2)
        leak_hits.extend(check_prompt_leaks(pid, system, t1, t2))
        c = r["condicao"]
        if c not in first_by_cond:
            first_by_cond[c] = r

    if leak_hits:
        errors.extend(leak_hits)

    checks["n_prompts_built"] = n_built
    checks["n_checksums"] = len(checksums)

    # Render policy: default first of C1–C4 only; --render-all writes all built
    rendered = 0
    render_targets: list[dict[str, str]] = []
    if render_all:
        render_targets = list(build_rows)
    else:
        for c in ("C1", "C2", "C3", "C4"):
            if c in first_by_cond:
                render_targets.append(first_by_cond[c])
            else:
                # from full base if filter excluded a condition
                for r in rows_all:
                    if r["condicao"] == c:
                        render_targets.append(r)
                        break

    if render_targets:
        RENDERED_DIR.mkdir(parents=True, exist_ok=True)
        for r in render_targets:
            pid = r["persona_id"]
            t1 = bp.build_turn1_user(stimuli, r)
            t2 = bp.build_turn2_user(stimuli, r)
            bp.write_triplet(RENDERED_DIR, pid, system, t1, t2)
            rendered += 3

    checks["rendered_files"] = rendered
    checks["render_all"] = render_all
    checks["rendered_persona_ids"] = [r["persona_id"] for r in render_targets]

    # Dry-run summary (checksums + meta) — avoid 2400 huge prompt files by default
    summary = {
        "datetime": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n_prompts_built": n_built,
        "protocolo_versao": proto,
        "stimuli_versao": stim_v,
        "schema_hash": checks.get("schema_hash"),
        "model": api_cfg.get("modelo"),
        "condicao_counts": checks.get("condicao_counts"),
        "locked_md5": md5_ok,
        "prompt_checksums_sha256": checksums,
        "rendered_files": rendered,
        "note": (
            "Dry-run N=800 — sem chamada API; checksums em vez de 2400 ficheiros "
            "de prompt (usar --render-all se necessário)."
        ),
    }
    DRY_RUN_SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Safety: never write into locked sources or pilot_outputs from this path
    checks["out_dir"] = str(OUT_DIR)
    checks["pilot_outputs_untouched"] = True

    dry_pass = len(errors) == 0
    _write_dry_validation(dry_pass, checks, errors, warnings)

    if dry_pass:
        print("DRY-RUN OK")
        print(f"  N={len(rows_all)} | conditions C1–C4 ×{PER_CELL} | protocolo={proto}")
        print(f"  schema_hash={checks.get('schema_hash')}")
        print(f"  prompts_built={n_built} | checksums → {DRY_RUN_SUMMARY}")
        print(f"  rendered={rendered} files → {RENDERED_DIR}")
        print(f"  report → {VALIDATION_JSON}")
        print("  locked MD5s OK")
        if warnings:
            for w in warnings:
                print(f"  WARN: {w}")
        return 0

    print("DRY-RUN FAIL", file=sys.stderr)
    for e in errors:
        print(f"  ERROR: {e}", file=sys.stderr)
    for w in warnings:
        print(f"  WARN: {w}", file=sys.stderr)
    return 1


def _write_dry_validation(
    dry_pass: bool,
    checks: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "dry-run",
        "collection_pass_technical": None,
        "dry_run_pass": dry_pass,
        "datetime": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n_expected": N_EXPECTED,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "note": (
            "Dry-run N=800 — NÃO é recolha nem teste de hipóteses; "
            "sem resultados empíricos; inputs LOCKED intocados."
        ),
    }
    VALIDATION_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_collection_report(
    stimuli: dict,
    api_cfg: dict[str, Any],
    all_base_ids: list[str],
) -> dict[str, Any]:
    """Build validation report from raw_responses.csv (full or partial run)."""
    rows: list[dict[str, str]] = []
    if RAW_CSV.exists():
        with open(RAW_CSV, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    by_id = {r["persona_id"]: r for r in rows}
    n_rows = len(rows)
    cond_counts = Counter(r.get("condicao", "") for r in rows)

    n_tech = sum(1 for r in rows if (r.get("technical_failure") or "").strip())
    n_valid = sum(
        1
        for r in rows
        if not (r.get("technical_failure") or "").strip()
        and r.get("turn1_parse_status") == "valid"
        and r.get("turn2_parse_status") == "valid"
    )
    n_format_invalid = sum(
        1
        for r in rows
        if not (r.get("technical_failure") or "").strip()
        and (
            r.get("turn1_parse_status") == "invalid"
            or r.get("turn2_parse_status") == "invalid"
        )
    )
    n_t1_retry = sum(1 for r in rows if str(r.get("turn1_retry_flag", "")) == "1")
    n_t2_retry = sum(1 for r in rows if str(r.get("turn2_retry_flag", "")) == "1")

    missing = [pid for pid in all_base_ids if pid not in by_id]
    extra = [pid for pid in by_id if pid not in set(all_base_ids)]

    # Completeness: all 800 present; tech failures flagged (not invented); no fake fills
    all_present = len(missing) == 0 and n_rows == N_EXPECTED and len(extra) == 0
    cells_ok = all(cond_counts.get(c, 0) == PER_CELL for c in ("C1", "C2", "C3", "C4"))
    schema_h = bp.schema_hash(stimuli)
    proto = stimuli.get("protocolo_versao", bp.PROTOCOLO_VERSAO)
    model = api_cfg.get("modelo", "")

    # Prefer zero tech failures; completeness still requires 800 rows + flags
    collection_pass = bool(
        all_present
        and cells_ok
        and n_tech == 0
        and len(extra) == 0
    )
    # Document: if tech fails >0 but all rows present and flagged, completeness_ok
    completeness_ok = bool(all_present and cells_ok)

    report = {
        "mode": "collection",
        "collection_pass_technical": collection_pass,
        "completeness_ok": completeness_ok,
        "datetime": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n_expected": N_EXPECTED,
        "n_rows": n_rows,
        "n_missing": len(missing),
        "missing_persona_ids_sample": missing[:20],
        "n_extra": len(extra),
        "condicao_counts": dict(sorted(cond_counts.items())),
        "n_valid": n_valid,
        "n_tech_fail": n_tech,
        "n_format_invalid": n_format_invalid,
        "n_turn1_retry": n_t1_retry,
        "n_turn2_retry": n_t2_retry,
        "schema_hash": schema_h,
        "model": model,
        "protocol_version": proto,
        "stimuli_version": stimuli.get("stimuli_versao", bp.STIMULI_VERSAO),
        "criteria": {
            "all_800_rows_present": all_present,
            "200_per_condition": cells_ok,
            "zero_tech_failures_preferred": n_tech == 0,
            "no_invented_fills": True,
            "note": (
                "collection_pass_technical exige 800 linhas, 200/célula e zero "
                "technical_failure. Format-invalid é auditado em n_format_invalid "
                "(não bloqueia pass se T1/T2 correram sem tech-fail). "
                "Análise/exclusões = passo separado."
            ),
        },
        "note": (
            "Recolha N=800 — sem inferência H1–H3; analysis_dataset é scaffold "
            "opcional após validação."
        ),
    }
    return report


def write_analysis_scaffold(stimuli: dict) -> None:
    """Optional scaffold AFTER validation — empty scores not invented; schema only."""
    fieldnames = [
        "persona_id",
        "condicao",
        "ciclo_id",
        "valencia_id",
        "BI1",
        "BI2",
        "BI3",
        "BI4",
        "MC_FRAME",
        "CE1",
        "CE2",
        "CE3",
        "turn1_parse_status",
        "turn2_parse_status",
        "technical_failure",
        "include_in_analysis",
    ]
    rows_out: list[dict[str, str]] = []
    if not RAW_CSV.exists():
        return
    with open(RAW_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            tech = (r.get("technical_failure") or "").strip()
            ok = (
                not tech
                and r.get("turn1_parse_status") == "valid"
                and r.get("turn2_parse_status") == "valid"
            )
            rows_out.append(
                {
                    "persona_id": r.get("persona_id", ""),
                    "condicao": r.get("condicao", ""),
                    "ciclo_id": r.get("ciclo_id", ""),
                    "valencia_id": r.get("valencia_id", ""),
                    "BI1": r.get("BI1", "") if ok else "",
                    "BI2": r.get("BI2", "") if ok else "",
                    "BI3": r.get("BI3", "") if ok else "",
                    "BI4": r.get("BI4", "") if ok else "",
                    "MC_FRAME": r.get("MC_FRAME", "") if ok else "",
                    "CE1": r.get("CE1", "") if ok else "",
                    "CE2": r.get("CE2", "") if ok else "",
                    "CE3": r.get("CE3", "") if ok else "",
                    "turn1_parse_status": r.get("turn1_parse_status", ""),
                    "turn2_parse_status": r.get("turn2_parse_status", ""),
                    "technical_failure": tech,
                    "include_in_analysis": "1" if ok else "0",
                }
            )
    with open(ANALYSIS_SCAFFOLD, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)


def run_live(
    *,
    only: str | None = None,
    limit: int | None = None,
    from_id: str | None = None,
    to_id: str | None = None,
    resume: bool = False,
    write_analysis: bool = False,
) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Erro: OPENAI_API_KEY não definida no ambiente. "
            "Exportar antes da recolha (nunca hardcoded).",
            file=sys.stderr,
        )
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "Erro: pacote openai não instalado. pip install openai",
            file=sys.stderr,
        )
        return 2

    # Verify locked MD5s before any write
    for name, expected in LOCKED_MD5.items():
        got = file_md5(ROOT / name)
        if got != expected:
            print(
                f"Erro: MD5 LOCKED alterado {name}: {got} != {expected}",
                file=sys.stderr,
            )
            return 2

    client = OpenAI(api_key=api_key, max_retries=0)

    stimuli = bp.load_stimuli()
    api_cfg = bp.get_api_config(stimuli)
    schemas = bp.get_schemas(stimuli)
    fieldnames = bp.audit_field_names(stimuli)

    rows_all = load_n800_base()
    all_ids = [r["persona_id"] for r in rows_all]
    rows = filter_rows(
        rows_all, only=only, limit=limit, from_id=from_id, to_id=to_id
    )

    if resume:
        done = load_completed_ids()
        before = len(rows)
        rows = [r for r in rows if r["persona_id"] not in done]
        print(f"Resume: saltando {before - len(rows)} já completos; restam {len(rows)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_API_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_CSV.exists():
        with open(RAW_CSV, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    results = []
    for row in rows:
        print(f"Running {row['persona_id']} ({row['condicao']}) …")
        rec = run_persona_live(client, stimuli, row, api_cfg, schemas, fieldnames)
        results.append(rec)
        tf = rec.get("technical_failure") or ""
        print(
            f"  turn1={rec.get('turn1_parse_status')} "
            f"turn2={rec.get('turn2_parse_status')} "
            f"tech_fail={bool(tf)}"
        )

    report = build_collection_report(stimuli, api_cfg, all_ids)
    report["n_processed_this_run"] = len(results)
    report["processed_persona_ids"] = [r.get("persona_id") for r in results]
    VALIDATION_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if write_analysis and report.get("completeness_ok"):
        write_analysis_scaffold(stimuli)
        print(f"analysis_dataset scaffold → {ANALYSIS_SCAFFOLD}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    # Exit 0 if this run had no new tech fails among processed; full pass in report
    run_tech = sum(1 for r in results if r.get("technical_failure"))
    if only or limit is not None or from_id or to_id or resume:
        # Partial run: success if no tech fail in this batch
        return 0 if run_tech == 0 else 1
    return 0 if report["collection_pass_technical"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recolha N=800 LLM v1.6 FINAL (sem alterar inputs LOCKED)."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validar 800 prompt builds + leaks (sem API).",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="Filtrar a uma persona (ex.: P037).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar a N personas (após filtros).",
    )
    ap.add_argument(
        "--from-id",
        default=None,
        help="Incluir a partir de persona_id (ex.: P100).",
    )
    ap.add_argument(
        "--to-id",
        default=None,
        help="Incluir até persona_id (ex.: P200).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Saltar persona_ids já completos em raw_responses.csv.",
    )
    ap.add_argument(
        "--render-all",
        action="store_true",
        help="Dry-run: escrever todos os triples de prompt (pesado).",
    )
    ap.add_argument(
        "--write-analysis-scaffold",
        action="store_true",
        help="Após recolha completa (completeness_ok), escrever analysis_dataset.csv.",
    )
    args = ap.parse_args()

    if args.dry_run:
        return run_dry_run(
            only=args.only,
            limit=args.limit,
            from_id=args.from_id,
            to_id=args.to_id,
            render_all=args.render_all,
        )
    return run_live(
        only=args.only,
        limit=args.limit,
        from_id=args.from_id,
        to_id=args.to_id,
        resume=args.resume,
        write_analysis=args.write_analysis_scaffold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
