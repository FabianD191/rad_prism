#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Consistency Check for RadPRISM Concept Predictions (optional post-processing)
================================================================================

An optional step that runs *after* the RadPRISM inference/evaluation: it sends
the per-concept classification decisions and the top retrieved text snippet to an
LLM and asks it to, per concept:

  1. assess the **consistency** between the binary classification decision and the
     meaning of the retrieved text (``consistent`` / ``inconsistent`` / ``uncertain``),
     with a short feedback note, and
  2. (optional) produce a cleaned / corrected **final text proposal**.

The consistency assessment is the core component; the final-text proposal is
optional and controlled by ``prompting.produce_final_text_proposal``.

Inputs per case (as produced by ``run_inference_demo.py`` and the evaluation
scripts):
  - ``classification.csv`` : columns ``concept`` and ``probability`` (or ``pred_prob``)
  - ``retrieval_top_matches.csv`` : long format ``concept, rank, sim, text``
  - a thresholds JSON with ``{concept: {thr_f1, thr_j}}`` (e.g. the checkpoint's
    ``radprism_val_thresholds.json``); the script binarizes probabilities with it.

The LLM output is validated against a guided JSON schema. Use ``--dry-run`` to
build and inspect the payload/schema/messages **without** calling any LLM (handy
for testing on the dummy outputs).

Usage:
  python evaluation/llm_consistency_check.py --config config/llm_consistency.yaml
  python evaluation/llm_consistency_check.py --config config/llm_consistency.yaml --dry-run

Requires an OpenAI-compatible chat endpoint (set api.api_base / api.api_key /
api.model_name in the config, or the OPENAI_API_KEY env var) for a real run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Default configuration (overridden by the YAML config and CLI flags)
# ---------------------------------------------------------------------------
DEFAULT_CFG: Dict[str, Any] = {
    "api": {
        "api_base": "http://your-llm-endpoint:port",
        "api_key": "",                       # or set OPENAI_API_KEY
        "model_name": "your-model-name",
    },
    "inputs": {
        "thresholds_json": "data/radprism_checkpoint/radprism_val_thresholds.json",
        "threshold_key": "thr_j",            # which threshold to binarize with
        "allow_threshold_fallback": False,
        "fallback_threshold_keys": ["thr_f1", "thr_j"],
        "default_threshold": 0.5,
        "retrieval_rank_to_use": 1,          # use the top-1 retrieved snippet per concept
        # Per-case filenames (as written by the inference demo / eval scripts).
        "concept_predictions_filename": "classification.csv",
        "retrieval_filename": "retrieval_top_matches.csv",
        # Single-case paths (used when batch.enabled is false).
        "concept_predictions_csv": "",
        "retrieval_csv": "",
    },
    "prompting": {
        "output_language": "en",
        "retrieved_text_language": "en",
        "produce_final_text_proposal": True,   # optional final cleaned-text proposal
        "system_prompt": (
            "You are a concept-level consistency assessment assistant for chest X-ray findings. "
            "Return ONLY valid JSON that strictly follows the provided schema. "
            "No markdown, no explanations, no extra keys."
        ),
        "task_instruction": (
            "For each concept, compare the binary classification decision with the meaning of the "
            "provided initial retrieved text snippet, and assign cls_text_consistency as consistent, "
            "inconsistent, or uncertain. "
            "If the initial_retrieved_text contains uncertain phrasing (e.g. 'cannot be excluded', "
            "'suspicion of', 'possibly', 'unclear', 'uncertain', 'not clearly assessable') or is "
            "otherwise ambiguous, use cls_text_consistency='uncertain'. "
            "Also use 'uncertain' when the retrieved text does not make sense for a chest X-ray "
            "report (e.g. describes findings outside the thorax such as head, legs, or pelvis) or "
            "clearly belongs to a different concept than the one it is attached to."
        ),
        "concept_semantics_instruction": (
            "binary_cls_label is already semantically resolved: 'present/not present' for "
            "support_devices and 'pathological/not pathological' for all other concepts."
        ),
        "output_field_instruction": (
            "For every concept leaf, fill these fields: binary_cls_label, initial_retrieved_text, "
            "cls_text_consistency, consistency_feedback (and final_text_proposal when requested). "
            "binary_cls_label and initial_retrieved_text must be copied exactly from the input."
        ),
        "additional_user_instructions": (
            "Use concise medical-style phrasing for final_text_proposal, based as closely as possible "
            "on initial_retrieved_text to keep it detailed. If the text contains comparative/temporal "
            "wording (e.g. 'still', 'known', 'compared to prior', 'increasing', 'regressing', "
            "'removed', 'explanted', or any date), reformulate it into a current-time statement without "
            "such wording. General pattern, e.g.: '<support device> removed on DD.MM.YYYY' => "
            "'<support device> not present'; 'Known <finding> still present' => '<finding> present'. "
            "The final_text_proposal must contain NO comparative/temporal wording. "
            "Keep consistency_feedback short and specific. If evidence is inconsistent, still base the "
            "final_text_proposal on the initial_retrieved_text unless overridden in "
            "inconsistency_priority_by_concept."
        ),
        # Per-concept override applied ONLY when cls_text_consistency='inconsistent':
        # base the final text on 'binary_cls_label' or on 'retrieved_text'.
        "inconsistency_priority_by_concept": {},
        "auto_build_overall_feedback": True,
    },
    "llm": {
        "use_json_schema": True,
        "fallback_to_json_object_on_400": False,
        "json_schema_name": "concept_consistency_output",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens_param": "max_tokens",
        "max_tokens": 5500,
        "timeout_s": 600.0,
        "max_retries": 5,
        "initial_backoff_s": 1.0,
        "max_backoff_s": 30.0,
        "retryable_statuses": [408, 429, 500, 502, 503, 504],
    },
    "output": {
        "write_artifacts": True,
        "raw_attempts_subdir": "raw_attempts",
        "prepared_payload_json": "prepared_payload.json",
        "prepared_schema_json": "output_schema.json",
        "messages_json": "messages.json",
        "raw_model_response_txt": "raw_model_response.txt",
        "final_model_output_json": "final_model_output.json",
        "run_summary_json": "run_summary.json",
        # Single-case output dir (batch mode writes into each case folder instead).
        "output_dir": "output/llm_consistency",
    },
    "batch": {
        "enabled": True,
        "root_dir": "output/inference_demo",
        "per_case_output_subdir": "llm_consistency",
        "batch_run_output_subdir": "llm_consistency_batch_runs",
        "max_cases": 0,                     # 0 => all discovered cases
        "continue_on_error": True,
    },
    "runtime": {
        "verbose": True,
        "dry_run": False,
    },
}

CONSISTENCY_ENUM = ["consistent", "inconsistent", "uncertain"]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config files: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def vlog(cfg: Dict[str, Any], msg: str) -> None:
    if bool(cfg["runtime"].get("verbose", True)):
        print(msg)


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_int(value: Any) -> Optional[int]:
    v = parse_float(value)
    return int(v) if v is not None else None


def load_json_object(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(obj)}")
    return obj


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str, data: Any) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def extract_message_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            txt = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if txt:
                parts.append(str(txt))
        return "\n".join(parts).strip()
    return str(content)


# ---------------------------------------------------------------------------
# Loading + preprocessing
# ---------------------------------------------------------------------------
def load_concept_predictions(path: str) -> List[Dict[str, Any]]:
    """Load per-concept probabilities. Accepts a 'probability' or 'pred_prob' column."""
    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in concept predictions CSV: {path}")
    if "concept" not in rows[0]:
        raise KeyError(f"Missing 'concept' column in {path}")
    prob_col = "probability" if "probability" in rows[0] else ("pred_prob" if "pred_prob" in rows[0] else None)
    if prob_col is None:
        raise KeyError(f"Missing a probability column ('probability' or 'pred_prob') in {path}")

    out, seen = [], set()
    for row in rows:
        concept = str(row.get("concept", "") or "").strip()
        if not concept or concept in seen:
            continue
        prob = parse_float(row.get(prob_col))
        if prob is None:
            raise ValueError(f"Invalid probability for concept '{concept}' in {path}")
        out.append({"concept_path": concept, "pred_prob": float(prob)})
        seen.add(concept)
    if not out:
        raise ValueError(f"No valid concept rows found in {path}")
    return out


def load_retrieval_top_matches(path: str, requested_rank: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Select one retrieved snippet per concept from a long-format CSV (concept, rank, sim, text)."""
    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in retrieval CSV: {path}")
    if not {"concept", "rank", "text"}.issubset(rows[0].keys()):
        raise ValueError(f"Retrieval CSV must have columns concept, rank, text (long format): {path}")

    candidates: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        concept = str(row.get("concept", "") or "").strip()
        text = str(row.get("text", "") or "").strip()
        if not concept or not text:
            continue
        candidates.setdefault(concept, []).append({
            "text": text,
            "rank": parse_int(row.get("rank")) or 10 ** 9,
            "sim": parse_float(row.get("sim")),
        })

    selected: Dict[str, Dict[str, Any]] = {}
    for concept, items in candidates.items():
        exact = [x for x in items if x["rank"] == requested_rank]
        pool = exact if exact else items
        best = sorted(pool, key=lambda x: (x["rank"], -(x["sim"] if x["sim"] is not None else -1.0)))[0]
        selected[concept] = best

    first = rows[0]
    meta = {"n_rows": len(rows), "n_selected_concepts": len(selected),
            "sample_id": str(first.get("sample_id", "") or "").strip()}
    return selected, meta


def choose_threshold_for_concept(concept: str, thresholds: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[float, str]:
    icfg = cfg["inputs"]
    key = str(icfg.get("threshold_key", "thr_j"))
    entry = thresholds.get(concept, {})
    if not isinstance(entry, dict):
        entry = {}
    v = parse_float(entry.get(key))
    if v is not None:
        return float(v), key
    if bool(icfg.get("allow_threshold_fallback", False)):
        for fk in icfg.get("fallback_threshold_keys", []):
            vv = parse_float(entry.get(fk))
            if vv is not None:
                return float(vv), fk
    return float(icfg.get("default_threshold", 0.5)), "default"


def concept_label_pair(concept: str) -> Tuple[str, str]:
    """Return (label_if_0, label_if_1) for a concept."""
    if concept.startswith("support_devices."):
        return "not present", "present"
    return "not pathological", "pathological"


def set_nested_value(root: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def get_nested_value(root: Dict[str, Any], dotted: str) -> Any:
    cur: Any = root
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def prepare_concept_evidence(concept_predictions, retrieval_top_map, thresholds, cfg):
    items, missing_retrieval, used_default = [], [], []
    for row in concept_predictions:
        concept = row["concept_path"]
        prob = float(row["pred_prob"])
        threshold, source = choose_threshold_for_concept(concept, thresholds, cfg)
        pred_bin = int(prob >= threshold)
        label_0, label_1 = concept_label_pair(concept)
        top = retrieval_top_map.get(concept) or {"text": "", "rank": None, "sim": None}
        if concept not in retrieval_top_map:
            missing_retrieval.append(concept)
        if source == "default":
            used_default.append(concept)
        items.append({
            "concept_path": concept, "pred_prob": prob,
            "threshold_used": threshold, "threshold_source": source,
            "binary_cls_prediction": pred_bin,
            "binary_cls_label": label_1 if pred_bin == 1 else label_0,
            "retrieved_text": str(top.get("text", "") or ""),
            "retrieved_rank": top.get("rank"), "retrieved_similarity": top.get("sim"),
        })
    items.sort(key=lambda x: x["concept_path"])
    stats = {"n_concepts": len(items), "n_missing_retrieval": len(missing_retrieval),
             "missing_retrieval_concepts": missing_retrieval,
             "n_default_threshold_used": len(used_default), "default_threshold_concepts": used_default}
    return items, stats


# ---------------------------------------------------------------------------
# Guided output schema
# ---------------------------------------------------------------------------
def _empty_obj() -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": {}, "required": []}


def leaf_field_names(include_final: bool) -> List[str]:
    fields = ["binary_cls_label", "initial_retrieved_text"]
    if include_final:
        fields.append("final_text_proposal")
    fields += ["cls_text_consistency", "consistency_feedback"]
    return fields


def make_leaf_schema(include_final: bool) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "binary_cls_label": {"type": "string"},
        "initial_retrieved_text": {"type": "string"},
    }
    if include_final:
        props["final_text_proposal"] = {"type": "string"}
    props["cls_text_consistency"] = {"type": "string", "enum": CONSISTENCY_ENUM}
    props["consistency_feedback"] = {"type": "string"}
    fields = leaf_field_names(include_final)
    return {"type": "object", "additionalProperties": False, "properties": props, "required": fields}


def build_output_schema(concept_paths: List[str], include_final: bool) -> Dict[str, Any]:
    concepts_schema = _empty_obj()
    for cp in concept_paths:
        cur = concepts_schema
        parts = cp.split(".")
        for p in parts[:-1]:
            if p not in cur["properties"]:
                cur["properties"][p] = _empty_obj()
                cur["required"].append(p)
            cur = cur["properties"][p]
        leaf = parts[-1]
        if leaf not in cur["properties"]:
            cur["properties"][leaf] = make_leaf_schema(include_final)
            cur["required"].append(leaf)
    return {
        "type": "object", "additionalProperties": False,
        "properties": {"overall_consistency_feedback": {"type": "string"}, "concepts": concepts_schema},
        "required": ["overall_consistency_feedback", "concepts"],
    }


# ---------------------------------------------------------------------------
# Prompt + messages
# ---------------------------------------------------------------------------
def build_payload(concept_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    concepts: Dict[str, Any] = {}
    for it in concept_items:
        set_nested_value(concepts, it["concept_path"], {
            "binary_cls_label": it["binary_cls_label"],
            "initial_retrieved_text": it["retrieved_text"],
        })
    return {"concepts": concepts}


def build_inconsistency_priority_instruction(cfg, concept_paths) -> str:
    raw = cfg["prompting"].get("inconsistency_priority_by_concept", {}) or {}
    valid = {str(k): str(v) for k, v in raw.items()
             if str(k) in concept_paths and str(v) in {"binary_cls_label", "retrieved_text", "equal"}}
    lines = [
        "If cls_text_consistency is inconsistent, default policy: base final_text_proposal on the initial_retrieved_text.",
        "If a conflict cannot be resolved confidently, use cautious wording and set cls_text_consistency='uncertain'.",
    ]
    if valid:
        lines.append("Concept-specific overrides (apply ONLY when cls_text_consistency='inconsistent'):")
        for cp in sorted(valid):
            lines.append(f"  {cp} = {valid[cp]}")
        lines.append("Override meaning: binary_cls_label = base final_text_proposal on the label only; "
                     "retrieved_text = base it on the retrieved text only. Keep cls_text_consistency='inconsistent'.")
    return "\n".join(lines)


def build_messages(payload, cfg, concept_paths, include_final) -> List[Dict[str, str]]:
    p = cfg["prompting"]
    user_lines = [
        str(p.get("task_instruction", "")).strip(),
        str(p.get("concept_semantics_instruction", "")).strip(),
        str(p.get("output_field_instruction", "")).strip(),
        f"Use this language for all generated text fields: {p.get('output_language', 'en')}.",
        "overall_consistency_feedback must summarize which concept paths were inconsistent "
        "(state clearly if none were).",
    ]
    if include_final:
        extra = str(p.get("additional_user_instructions", "")).strip()
        if extra:
            user_lines.append(extra)
        user_lines.append(build_inconsistency_priority_instruction(cfg, concept_paths))
    else:
        user_lines.append("Do NOT produce a final_text_proposal field; only assess consistency.")
    user_lines.append("Input evidence JSON:")
    user_lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
    return [
        {"role": "system", "content": str(p.get("system_prompt", "")).strip()},
        {"role": "user", "content": "\n\n".join(x for x in user_lines if x)},
    ]


def collect_concept_statuses(concepts_obj, prefix="") -> List[Tuple[str, str]]:
    rows = []
    if not isinstance(concepts_obj, dict):
        return rows
    for key, value in concepts_obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "cls_text_consistency" in value:
            rows.append((path, str(value.get("cls_text_consistency", "")).strip()))
        elif isinstance(value, dict):
            rows.extend(collect_concept_statuses(value, path))
    return rows


def build_overall_feedback_text(final_output: Dict[str, Any]) -> str:
    statuses = collect_concept_statuses(final_output.get("concepts", {}))
    inconsistent = sorted(cp for cp, st in statuses if st == "inconsistent")
    uncertain = sorted(cp for cp, st in statuses if st == "uncertain")
    if inconsistent:
        text = "Inconsistencies were found for: " + ", ".join(inconsistent) + "."
    else:
        text = "No inconsistencies were found between classification labels and retrieved texts."
    if uncertain:
        text += " Uncertain concepts: " + ", ".join(uncertain) + "."
    return text


# ---------------------------------------------------------------------------
# LLM call + validation
# ---------------------------------------------------------------------------
def build_expected_leaf_map(concept_items) -> Dict[str, Dict[str, str]]:
    return {it["concept_path"]: {"binary_cls_label": str(it["binary_cls_label"]),
                                 "initial_retrieved_text": str(it["retrieved_text"])}
            for it in concept_items}


def validate_output_structure(data, concept_paths, expected_leaf_map, include_final) -> None:
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object.")
    for key in ("overall_consistency_feedback", "concepts"):
        if key not in data:
            raise ValueError(f"Missing top-level key: {key}")
    concepts_obj = data.get("concepts")
    if not isinstance(concepts_obj, dict):
        raise ValueError("'concepts' must be an object.")
    required = leaf_field_names(include_final)
    for cp in concept_paths:
        leaf = get_nested_value(concepts_obj, cp)
        if not isinstance(leaf, dict):
            raise ValueError(f"Missing concept leaf in output: {cp}")
        for f in required:
            if f not in leaf:
                raise ValueError(f"Missing field '{f}' for concept '{cp}'")
        if str(leaf["cls_text_consistency"]) not in CONSISTENCY_ENUM:
            raise ValueError(f"Invalid cls_text_consistency for '{cp}'")
        exp = expected_leaf_map[cp]
        if str(leaf["binary_cls_label"]) != exp["binary_cls_label"]:
            raise ValueError(f"binary_cls_label mismatch for '{cp}' (must be copied from input).")
        if str(leaf["initial_retrieved_text"]).strip() != exp["initial_retrieved_text"].strip():
            raise ValueError(f"initial_retrieved_text mismatch for '{cp}' (must be copied from input).")


def build_openai_client(cfg: Dict[str, Any]):
    import httpx
    from openai import AsyncOpenAI
    api = cfg["api"]
    api_key = str(api.get("api_key", "") or "").strip() or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("No API key configured. Set api.api_key or the OPENAI_API_KEY env var.")
    timeout_s = float(cfg["llm"].get("timeout_s", 600.0))
    timeout = httpx.Timeout(connect=min(10.0, timeout_s), read=timeout_s,
                            write=min(30.0, timeout_s), pool=min(10.0, timeout_s))
    http_client = httpx.AsyncClient(timeout=timeout,
                                    limits=httpx.Limits(max_connections=8, max_keepalive_connections=8))
    return AsyncOpenAI(base_url=str(api["api_base"]), api_key=api_key, timeout=timeout_s,
                       http_client=http_client, max_retries=0)


def build_chat_kwargs(messages, schema, mode, cfg) -> Dict[str, Any]:
    llm = cfg["llm"]
    kwargs = {"model": cfg["api"]["model_name"], "messages": messages,
              "temperature": float(llm.get("temperature", 0.0)), "top_p": float(llm.get("top_p", 1.0)),
              "stream": False}
    kwargs[str(llm.get("max_tokens_param", "max_tokens"))] = int(llm.get("max_tokens", 5500))
    if mode == "json_schema":
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {
            "name": str(llm.get("json_schema_name", "concept_consistency_output")),
            "schema": schema, "strict": True}}
    else:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


async def call_model_with_retries(client, messages, schema, concept_paths, expected_leaf_map, include_final, cfg):
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
    llm = cfg["llm"]
    max_retries = int(llm.get("max_retries", 5))
    retryable = set(llm.get("retryable_statuses", [408, 429, 500, 502, 503, 504]))
    backoff = float(llm.get("initial_backoff_s", 1.0))
    max_backoff = float(llm.get("max_backoff_s", 30.0))
    use_schema = bool(llm.get("use_json_schema", True))
    allow_fallback = bool(llm.get("fallback_to_json_object_on_400", False))
    working = deepcopy(messages)

    for attempt in range(1, max_retries + 1):
        mode = "json_schema" if use_schema else "json_object"
        vlog(cfg, f"[LLM] attempt {attempt}/{max_retries} mode={mode}")
        try:
            t0 = time.perf_counter()
            resp = await asyncio.wait_for(client.chat.completions.create(**build_chat_kwargs(working, schema, mode, cfg)),
                                          timeout=float(llm.get("timeout_s", 600.0)))
            elapsed = time.perf_counter() - t0
            raw = extract_message_content_text(resp.choices[0].message.content)
            if not raw.strip():
                raise ValueError("LLM returned empty content")
            data = json.loads(raw)
            validate_output_structure(data, concept_paths, expected_leaf_map, include_final)
            return data, raw, mode, attempt, elapsed
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status == 400 and use_schema and allow_fallback:
                use_schema = False
                continue
            if status in retryable and attempt < max_retries:
                await asyncio.sleep(min(backoff, max_backoff)); backoff = min(backoff * 2, max_backoff)
                continue
            raise
        except (RateLimitError, APITimeoutError, APIConnectionError, asyncio.TimeoutError):
            if attempt < max_retries:
                await asyncio.sleep(min(backoff, max_backoff)); backoff = min(backoff * 2, max_backoff)
                continue
            raise
        except (json.JSONDecodeError, ValueError) as exc:
            if isinstance(exc, json.JSONDecodeError) and use_schema and allow_fallback:
                use_schema = False
            if attempt < max_retries:
                working = working + [{"role": "user", "content":
                    "Your previous answer was invalid or did not follow the schema. "
                    "Return ONLY valid JSON following the required schema exactly."}]
                await asyncio.sleep(min(backoff, max_backoff)); backoff = min(backoff * 1.6, max_backoff)
                continue
            raise
    raise RuntimeError("Model call failed after max retries.")


# ---------------------------------------------------------------------------
# Single-case + batch pipeline
# ---------------------------------------------------------------------------
async def run_single_case(cfg, client=None, case_label="") -> Dict[str, Any]:
    t0 = time.perf_counter()
    prefix = f"[{case_label}] " if case_label else ""
    icfg, ocfg = cfg["inputs"], cfg["output"]
    concept_csv = str(icfg["concept_predictions_csv"])
    retrieval_csv = str(icfg["retrieval_csv"])
    out_dir = str(ocfg["output_dir"])
    include_final = bool(cfg["prompting"].get("produce_final_text_proposal", True))

    concept_predictions = load_concept_predictions(concept_csv)
    thresholds = load_json_object(str(icfg["thresholds_json"]))
    retrieval_map, retrieval_meta = load_retrieval_top_matches(retrieval_csv, int(icfg.get("retrieval_rank_to_use", 1)))
    concept_items, prep_stats = prepare_concept_evidence(concept_predictions, retrieval_map, thresholds, cfg)

    concept_paths = [x["concept_path"] for x in concept_items]
    schema = build_output_schema(concept_paths, include_final)
    expected_leaf_map = build_expected_leaf_map(concept_items)
    payload = build_payload(concept_items)
    messages = build_messages(payload, cfg, concept_paths, include_final)
    sample_id = retrieval_meta.get("sample_id") or Path(retrieval_csv).parent.name

    vlog(cfg, f"{prefix}{len(concept_items)} concepts | consistency"
              + (" + final_text_proposal" if include_final else " only")
              + f" | out={out_dir}")

    if bool(ocfg.get("write_artifacts", True)):
        ensure_dir(out_dir)
        write_json(str(Path(out_dir) / ocfg["prepared_payload_json"]), payload)
        write_json(str(Path(out_dir) / ocfg["prepared_schema_json"]), schema)
        write_json(str(Path(out_dir) / ocfg["messages_json"]), {"messages": messages})

    run_summary = {"status": "dry_run" if cfg["runtime"].get("dry_run") else "success",
                   "sample_id": sample_id, "concept_predictions_csv": concept_csv,
                   "retrieval_csv": retrieval_csv, "n_concepts": len(concept_items),
                   "preprocessing_stats": prep_stats, "output_dir": out_dir}

    if bool(cfg["runtime"].get("dry_run", False)):
        run_summary["total_elapsed_s"] = round(time.perf_counter() - t0, 3)
        if bool(ocfg.get("write_artifacts", True)):
            write_json(str(Path(out_dir) / ocfg["run_summary_json"]), run_summary)
        vlog(cfg, f"{prefix}Dry-run: prepared payload/schema/messages, skipped LLM call.")
        return run_summary

    own_client = client is None
    active = client or build_openai_client(cfg)
    try:
        final_output, raw, used_mode, attempts, model_elapsed = await call_model_with_retries(
            active, messages, schema, concept_paths, expected_leaf_map, include_final, cfg)
    finally:
        if own_client:
            await active.close()

    if bool(cfg["prompting"].get("auto_build_overall_feedback", True)):
        final_output["overall_consistency_feedback"] = build_overall_feedback_text(final_output)

    if bool(ocfg.get("write_artifacts", True)):
        write_text(str(Path(out_dir) / ocfg["raw_model_response_txt"]), raw)
        write_json(str(Path(out_dir) / ocfg["final_model_output_json"]), final_output)
    run_summary.update({"used_response_mode": used_mode, "attempts": attempts,
                        "model_elapsed_s": round(model_elapsed, 3),
                        "total_elapsed_s": round(time.perf_counter() - t0, 3)})
    if bool(ocfg.get("write_artifacts", True)):
        write_json(str(Path(out_dir) / ocfg["run_summary_json"]), run_summary)
    vlog(cfg, f"{prefix}done ({run_summary['status']}).")
    return run_summary


def discover_batch_cases(cfg) -> List[Dict[str, str]]:
    bcfg, icfg = cfg["batch"], cfg["inputs"]
    root = Path(str(bcfg.get("root_dir", ""))).resolve()
    if not root.exists():
        raise FileNotFoundError(f"batch.root_dir not found: {root}")
    concept_fn = str(icfg.get("concept_predictions_filename", "classification.csv"))
    retrieval_fn = str(icfg.get("retrieval_filename", "retrieval_top_matches.csv"))
    cases, seen = [], set()
    for cpath in root.rglob(concept_fn):
        case_dir = cpath.parent.resolve()
        rpath = case_dir / retrieval_fn
        if not rpath.is_file() or str(case_dir) in seen:
            continue
        seen.add(str(case_dir))
        cases.append({"case_dir": str(case_dir), "concept_predictions_csv": str(cpath), "retrieval_csv": str(rpath)})
    cases.sort(key=lambda x: x["case_dir"])
    max_cases = int(bcfg.get("max_cases", 0))
    return cases[:max_cases] if max_cases > 0 else cases


async def run_batch(cfg) -> None:
    t_start = time.perf_counter()
    bcfg = cfg["batch"]
    cases = discover_batch_cases(cfg)
    if not cases:
        raise RuntimeError(f"No cases found under {bcfg.get('root_dir')} "
                           f"(need {cfg['inputs']['concept_predictions_filename']} + "
                           f"{cfg['inputs']['retrieval_filename']}).")
    root = Path(str(bcfg["root_dir"])).resolve()
    run_id = make_run_id()
    batch_dir = root / str(bcfg.get("batch_run_output_subdir", "llm_consistency_batch_runs")) / run_id
    ensure_dir(str(batch_dir))
    vlog(cfg, f"=== Batch: {len(cases)} cases under {root} ===")

    dry = bool(cfg["runtime"].get("dry_run", False))
    client = None if dry else build_openai_client(cfg)
    results = []
    try:
        for i, case in enumerate(cases, 1):
            case_dir = Path(case["case_dir"])
            case_cfg = deepcopy(cfg)
            case_cfg["batch"]["enabled"] = False
            case_cfg["inputs"]["concept_predictions_csv"] = case["concept_predictions_csv"]
            case_cfg["inputs"]["retrieval_csv"] = case["retrieval_csv"]
            case_cfg["output"]["output_dir"] = str(case_dir / str(bcfg.get("per_case_output_subdir", "llm_consistency")))
            label = f"{i}/{len(cases)} {case_dir.name}"
            try:
                summary = await run_single_case(case_cfg, client=client, case_label=label)
                summary["case_dir"] = str(case_dir)
                results.append(summary)
            except Exception as exc:
                results.append({"status": "failed", "case_dir": str(case_dir), "error": repr(exc)})
                vlog(cfg, f"[{label}] ERROR: {exc}")
                if not bool(bcfg.get("continue_on_error", True)):
                    raise
    finally:
        if client is not None:
            await client.close()

    summary = {"root_dir": str(root), "run_id": run_id, "n_cases": len(cases),
               "n_success": sum(1 for r in results if r.get("status") == "success"),
               "n_dry_run": sum(1 for r in results if r.get("status") == "dry_run"),
               "n_failed": sum(1 for r in results if r.get("status") == "failed"),
               "total_elapsed_s": round(time.perf_counter() - t_start, 3), "cases": results}
    write_json(str(batch_dir / "batch_run_summary.json"), summary)
    vlog(cfg, f"=== Batch done: {summary['n_success']} ok, {summary['n_dry_run']} dry-run, "
              f"{summary['n_failed']} failed -> {batch_dir} ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM cls/retrieval consistency check for RadPRISM outputs.")
    ap.add_argument("--config", required=True, help="Path to config/llm_consistency.yaml")
    ap.add_argument("--dry-run", action="store_true", help="Build payload/schema/messages without calling the LLM.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--api-base", type=str, default=None)
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--thresholds-json", type=str, default=None)
    ap.add_argument("--batch-root-dir", type=str, default=None)
    ap.add_argument("--batch-max-cases", type=int, default=None)
    ap.add_argument("--concept-predictions-csv", type=str, default=None, help="Single-case mode.")
    ap.add_argument("--retrieval-csv", type=str, default=None, help="Single-case mode.")
    ap.add_argument("--output-dir", type=str, default=None)
    return ap.parse_args()


def apply_cli(cfg: Dict[str, Any], a: argparse.Namespace) -> Dict[str, Any]:
    cfg = deepcopy(cfg)
    if a.dry_run:
        cfg["runtime"]["dry_run"] = True
    if a.quiet:
        cfg["runtime"]["verbose"] = False
    if a.api_base:
        cfg["api"]["api_base"] = a.api_base
    if a.api_key:
        cfg["api"]["api_key"] = a.api_key
    if a.model:
        cfg["api"]["model_name"] = a.model
    if a.thresholds_json:
        cfg["inputs"]["thresholds_json"] = a.thresholds_json
    if a.batch_root_dir:
        cfg["batch"]["enabled"] = True
        cfg["batch"]["root_dir"] = a.batch_root_dir
    if a.batch_max_cases is not None:
        cfg["batch"]["max_cases"] = int(a.batch_max_cases)
    if a.concept_predictions_csv:
        cfg["batch"]["enabled"] = False
        cfg["inputs"]["concept_predictions_csv"] = a.concept_predictions_csv
    if a.retrieval_csv:
        cfg["inputs"]["retrieval_csv"] = a.retrieval_csv
    if a.output_dir:
        cfg["output"]["output_dir"] = a.output_dir
    return cfg


def main() -> None:
    args = parse_args()
    cfg = deep_merge(DEFAULT_CFG, load_yaml(args.config))
    cfg = apply_cli(cfg, args)
    if bool(cfg["batch"].get("enabled", False)):
        asyncio.run(run_batch(cfg))
    else:
        asyncio.run(run_single_case(cfg))


if __name__ == "__main__":
    main()
