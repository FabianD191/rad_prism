#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based Radiology Report Structuring and Labeling
====================================================

Processes free-text radiology reports through an LLM API (OpenAI-compatible)
to extract structured JSON with text spans and labels according to a
user-defined schema.

Key features:
  - Schema and validation models are loaded from an external JSON file,
    so the same script works for any report type without code changes.
  - System-prompt instructions are loaded from a separate Markdown file.
  - Optional YAML config file for setting defaults (CLI args override).
  - Async processing with configurable concurrency, retries, and backoff.
  - Guided JSON generation (json_schema response_format) with automatic
    fallback to plain json_object mode when the provider does not support it.
  - Per-report debug dumps on failure for later inspection.

Usage:
  python report_structuring_labeling.py --config config.yaml
  or configure via CLI flags (see --help).
"""

from __future__ import annotations

import os
import json
import time
import random
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Optional, Any

import pandas as pd
from pydantic import ValidationError

from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)
import httpx


# ═══════════════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("report_struct_label")


# ═══════════════════════════════════════════════════════════════════════════════
# YAML config loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    The YAML keys should match CLI argument names (with underscores, not
    hyphens). For example, 'report_text_column: my_col' in YAML corresponds
    to --report-text-column on the CLI.

    Raises ImportError if PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config files: pip install pyyaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


# ═══════════════════════════════════════════════════════════════════════════════
# Schema loading — build Pydantic validator + inlined JSON schema from file
# ═══════════════════════════════════════════════════════════════════════════════

def load_json_schema(schema_path: str) -> dict:
    """
    Load the JSON schema that defines the expected structured output.

    The schema file should be a standard JSON Schema object (no $ref/$defs)
    describing the full output structure including all required fields, nested
    objects, enums, and constraints.

    Returns:
        dict: The parsed JSON schema.
    """
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    log.info("Loaded JSON schema from %s", schema_path)
    return schema


def build_pydantic_validator(schema: dict):
    """
    Dynamically build a Pydantic model from a JSON schema dict for runtime
    validation of LLM outputs.

    This uses pydantic's ability to validate arbitrary dicts against a schema
    without needing a hand-written model class. The returned callable accepts
    a dict and raises ValidationError if it does not conform.

    Returns:
        callable: A function that takes a dict and returns the validated dict,
                  raising pydantic.ValidationError on mismatch.
    """
    from pydantic import TypeAdapter
    adapter = TypeAdapter(dict)  # basic dict validation

    def validate(data: dict) -> dict:
        """
        Validate data against the loaded schema.
        Uses jsonschema for structural validation and returns the data as-is
        if valid (the schema itself enforces types, enums, required keys, etc.).
        """
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
        return data

    return validate


# ═══════════════════════════════════════════════════════════════════════════════
# Custom exceptions and timing helpers
# ═══════════════════════════════════════════════════════════════════════════════

class EmptyModelContentError(RuntimeError):
    """Raised when the LLM returns an empty or null message content."""
    def __init__(self, msg: str, raw: str | None = None, resp_dump: dict | None = None):
        super().__init__(msg)
        self.raw = raw
        self.resp_dump = resp_dump


class CallTiming:
    """Tracks timing statistics for a single LLM call (possibly with retries)."""
    __slots__ = ("attempts", "model_s_total", "model_s_final")

    def __init__(self, attempts: int, model_s_total: float, model_s_final: float):
        self.attempts = attempts
        self.model_s_total = model_s_total      # sum of all HTTP request durations
        self.model_s_final = model_s_final      # duration of the successful request


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════════

def load_completed_report_ids(output_dir: str) -> set[str]:
    """Scan output directory to find already-processed report IDs.

    Expects output files named like <report_id>_<suffix>.json.
    Debug files (ending in _debug.json) are excluded.
    """
    p = Path(output_dir)
    if not p.exists():
        return set()
    return {f.name.split("_", 1)[0] for f in p.glob("*.json") if not f.name.endswith("_debug.json")}


def as_str(val: Any) -> str:
    """Safely convert a value to string, returning '' for NaN/None."""
    return "" if pd.isna(val) else str(val)


def save_debug_case(
    debug_dir: str,
    report_id: str,
    error: str,
    messages: list[dict],
    raw: str | None = None,
    resp_dump: dict | None = None,
):
    """
    Save a debug JSON file when processing a report fails.

    Includes the report ID, error details, the full message history
    sent to the LLM, and optionally the raw response for post-hoc analysis.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = {
        "report_id": report_id,
        "timestamp": ts,
        "error": error,
        "messages": messages,
        "raw_response": raw,
        "response_dump": resp_dump,
    }
    path = Path(debug_dir) / f"{report_id}_{ts}_debug.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


async def backoff_sleep(backoff_s: float, max_backoff_s: float, max_jitter_s: float):
    """Sleep with exponential backoff and random jitter."""
    await asyncio.sleep(min(backoff_s, max_backoff_s) + random.uniform(0.0, max_jitter_s))


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI client factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_openai_client(
    api_base: str,
    api_key: str,
    max_concurrency: int,
    request_timeout_s: float,
) -> AsyncOpenAI:
    """
    Create an AsyncOpenAI client with custom timeout and connection pool settings.

    Internal retries are disabled (max_retries=0) to avoid double retry loops
    with the application-level retry logic.
    """
    timeout = httpx.Timeout(
        connect=10.0,
        read=request_timeout_s,
        write=10.0,
        pool=10.0,
    )
    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=max_concurrency,
        keepalive_expiry=30.0,
    )
    http_client = httpx.AsyncClient(timeout=timeout, limits=limits)

    return AsyncOpenAI(
        base_url=api_base,
        api_key=api_key,
        max_retries=0,
        http_client=http_client,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LLM call builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_chat_kwargs(
    messages: list[dict],
    model_name: str,
    json_schema: dict,
    use_guided_schema: bool,
    include_optional_params: bool,
    reasoning_effort_value: Optional[str],
    service_tier: Optional[str],
    max_tokens: Optional[int],
    request_timeout_s: float,
    temperature: float = 0.0,
) -> dict:
    """
    Build the keyword arguments dict for a chat completions API call.

    Supports two response_format modes:
      - json_schema: strict guided generation (preferred, if provider supports it)
      - json_object: plain JSON mode (fallback)

    Optional params like service_tier and reasoning_effort are only included
    when include_optional_params=True (they may cause 400 errors on some providers).
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
        "timeout": request_timeout_s,
        "stream": False,
    }

    if use_guided_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "report_structuring_and_labeling",
                "schema": json_schema,
                "strict": True,
            },
        }
    else:
        kwargs["response_format"] = {"type": "json_object"}

    if include_optional_params:
        if service_tier:
            kwargs["service_tier"] = service_tier
        if reasoning_effort_value:
            kwargs["reasoning_effort"] = reasoning_effort_value
            kwargs["extra_body"] = {"allowed_openai_params": ["reasoning_effort"]}

    return kwargs


# ═══════════════════════════════════════════════════════════════════════════════
# Model call with retries + validation
# ═══════════════════════════════════════════════════════════════════════════════

async def call_model_struct_label(
    client: AsyncOpenAI,
    system_prompt: str,
    user_input_text: str,
    *,
    model_name: str,
    json_schema: dict,
    validate_fn,
    max_retries: int,
    initial_backoff_s: float,
    max_backoff_s: float,
    max_jitter_s: float,
    retryable_statuses: set[int],
    service_tier: Optional[str],
    reasoning_effort_default: Optional[str],
    reasoning_effort_on_retry: Optional[str],
    max_tokens: Optional[int],
    request_timeout_s: float,
    temperature: float = 0.0,
) -> tuple[dict, CallTiming]:
    """
    Send a report to the LLM and validate the structured output.

    Implements:
      - Exponential backoff with jitter on retryable errors
      - Automatic fallback: guided schema -> json_object if 400 errors
      - On validation errors: appends corrective user message, increases
        reasoning effort, and retries
      - Returns (validated_dict, timing) on success; raises on exhaustion

    Args:
        client: AsyncOpenAI client instance
        system_prompt: Full system prompt text
        user_input_text: The formatted user message containing the report
        model_name: LLM model identifier
        json_schema: Inlined JSON schema for guided generation
        validate_fn: Callable that validates parsed JSON against the schema
        max_retries: Maximum number of retry attempts
        initial_backoff_s: Starting backoff duration in seconds
        max_backoff_s: Maximum backoff cap
        max_jitter_s: Maximum random jitter added to backoff
        retryable_statuses: HTTP status codes that trigger automatic retry
        service_tier: Optional service tier parameter
        reasoning_effort_default: Default reasoning effort level
        reasoning_effort_on_retry: Elevated reasoning effort for retries
        max_tokens: Maximum generation tokens
        request_timeout_s: Per-request timeout in seconds
        temperature: Sampling temperature (default 0 for deterministic output)

    Returns:
        Tuple of (validated output dict, CallTiming with attempt statistics)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input_text},
    ]

    backoff = initial_backoff_s
    guided_ok = True
    optional_params_ok = True
    reasoning_effort_current = reasoning_effort_default

    model_s_total = 0.0
    model_s_final = 0.0

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = _build_chat_kwargs(
                messages,
                model_name=model_name,
                json_schema=json_schema,
                use_guided_schema=guided_ok,
                include_optional_params=optional_params_ok,
                reasoning_effort_value=reasoning_effort_current,
                service_tier=service_tier,
                max_tokens=max_tokens,
                request_timeout_s=request_timeout_s,
                temperature=temperature,
            )

            # Time the HTTP request
            t_req0 = time.perf_counter()
            try:
                resp = await client.chat.completions.create(**kwargs)
            finally:
                dt_req = time.perf_counter() - t_req0
                model_s_total += dt_req

            model_s_final = dt_req

            # Extract and validate response content
            raw = resp.choices[0].message.content
            if raw is None or raw.strip() == "":
                raise EmptyModelContentError(
                    "message.content is empty/null.",
                    raw=None,
                    resp_dump=(resp.model_dump() if hasattr(resp, "model_dump") else {"repr": repr(resp)}),
                )

            data = json.loads(raw)
            validated = validate_fn(data)
            return validated, CallTiming(
                attempts=attempt,
                model_s_total=model_s_total,
                model_s_final=model_s_final,
            )

        except APIStatusError as e:
            status = getattr(e, "status_code", None)

            # Retryable server errors (429, 500, 502, 503, 504, etc.)
            if status in retryable_statuses:
                if attempt < max_retries:
                    await backoff_sleep(backoff, max_backoff_s, max_jitter_s)
                    backoff = min(backoff * 2.0, max_backoff_s)
                    continue
                raise

            # 400 Bad Request — progressively disable optional features
            if status == 400:
                if optional_params_ok:
                    optional_params_ok = False
                elif guided_ok:
                    guided_ok = False
                else:
                    raise

                if attempt < max_retries:
                    await backoff_sleep(backoff, max_backoff_s, max_jitter_s)
                    backoff = min(backoff * 1.6, max_backoff_s)
                    continue
                raise

            raise

        except (RateLimitError, APITimeoutError, APIConnectionError, EmptyModelContentError):
            if attempt < max_retries:
                await backoff_sleep(backoff, max_backoff_s, max_jitter_s)
                backoff = min(backoff * 2.0, max_backoff_s)
                continue
            raise

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            if isinstance(e, (json.JSONDecodeError, ValidationError)) or \
               (hasattr(e, '__module__') and 'jsonschema' in str(type(e))):
                # Append corrective message and increase reasoning effort
                messages = messages + [{
                    "role": "user",
                    "content": (
                        "Your last output was invalid or did not strictly match the schema.\n"
                        f"Error: {e}\n"
                        "Please return only valid JSON according to the schema "
                        "(no additional keys, no explanations)."
                    ),
                }]
                reasoning_effort_current = reasoning_effort_on_retry
                log.warning("Validation error on attempt %d: %s — retrying with higher effort", attempt, e)

                if attempt < max_retries:
                    await backoff_sleep(backoff, max_backoff_s, max_jitter_s)
                    backoff = min(backoff * 1.6, max_backoff_s)
                    continue
                raise
            raise

    raise RuntimeError("Model call failed after all retries.")


# ═══════════════════════════════════════════════════════════════════════════════
# Row processor (parallel-safe)
# ═══════════════════════════════════════════════════════════════════════════════

async def process_one_row(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    system_prompt: str,
    row: pd.Series,
    completed_set: set[str],
    completed_lock: asyncio.Lock,
    *,
    config: argparse.Namespace,
    json_schema: dict,
    validate_fn,
    report_text_column: str,
    exam_column: Optional[str],
    report_id_column: str,
    output_suffix: str,
    user_message_template: str,
    metadata_columns: list[str],
):
    """
    Process a single report row: call the LLM, validate, and save the result.

    This function is designed to run concurrently under an asyncio.Semaphore
    that limits parallelism. It handles:
      - Skip logic for already-processed report IDs (disk + in-memory set)
      - Building the user message from configured columns
      - Saving validated output as JSON
      - Saving debug dumps on failure
    """
    report_id = str(row[report_id_column])
    outfile = os.path.join(config.output_dir, f"{report_id}_{output_suffix}.json")

    # Skip if already exists on disk
    if os.path.exists(outfile):
        return

    async with sem:
        # Skip if completed in-memory (by another concurrent task)
        async with completed_lock:
            if report_id in completed_set:
                return

        # Build metadata: always include report_id, plus any extra columns
        metadata = {"report_id": report_id}
        for col in metadata_columns:
            if col != report_id_column:
                metadata[col] = as_str(row.get(col, ""))

        # Build user message from template
        template_values = {
            "report_text": row.get(report_text_column, "") or "",
        }
        if exam_column:
            template_values["exam_description"] = row.get(exam_column, "") or ""

        user_input = user_message_template.format(**template_values)

        messages_for_debug = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        t0_total = time.perf_counter()
        try:
            validated, timing = await call_model_struct_label(
                client=client,
                system_prompt=system_prompt,
                user_input_text=user_input,
                model_name=config.model_name,
                json_schema=json_schema,
                validate_fn=validate_fn,
                max_retries=config.max_retries,
                initial_backoff_s=config.initial_backoff_s,
                max_backoff_s=config.max_backoff_s,
                max_jitter_s=config.max_jitter_s,
                retryable_statuses=set(config.retryable_statuses),
                service_tier=config.service_tier,
                reasoning_effort_default=config.reasoning_effort_default,
                reasoning_effort_on_retry=config.reasoning_effort_on_retry,
                max_tokens=config.max_tokens,
                request_timeout_s=config.request_timeout_s,
                temperature=config.temperature,
            )

            # Simplified output structure: report_id + data
            final_output = {"report_id": report_id, "data": validated}

            # Atomic write via temp file + rename
            tmpfile = outfile + ".tmp"
            with open(tmpfile, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
            os.replace(tmpfile, outfile)

            async with completed_lock:
                completed_set.add(report_id)

            total_s = time.perf_counter() - t0_total
            non_model_s = max(0.0, total_s - timing.model_s_total)

            log.info(
                "[%s] saved | total=%.2fs | model_total=%.2fs | model_final=%.2fs | "
                "non_model=%.2fs | attempts=%d",
                report_id, total_s, timing.model_s_total, timing.model_s_final,
                non_model_s, timing.attempts,
            )

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            raw = None
            resp_dump = None
            if isinstance(e, EmptyModelContentError):
                raw = e.raw
                resp_dump = e.resp_dump
            save_debug_case(
                debug_dir=config.debug_dir,
                report_id=report_id,
                error=err,
                messages=messages_for_debug,
                raw=raw,
                resp_dump=resp_dump,
            )
            log.exception("[%s] failed (debug saved)", report_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Main processing loop
# ═══════════════════════════════════════════════════════════════════════════════

async def main_async(config: argparse.Namespace):
    """
    Main async entry point: loads data, creates tasks, and processes reports.

    Steps:
      1. Load system prompt and JSON schema
      2. Build Pydantic validator from schema
      3. Load report CSV
      4. Skip already-completed report IDs
      5. Launch concurrent processing tasks with semaphore-based throttling
    """
    scheduled_total = 0
    completed = load_completed_report_ids(config.output_dir)
    log.info("Found %d already-completed report IDs", len(completed))

    # Load system prompt
    system_prompt = Path(config.system_prompt_path).read_text(encoding="utf-8")

    # Load and prepare schema
    json_schema = load_json_schema(config.schema_path)
    validate_fn = build_pydantic_validator(json_schema)

    # Resolve API key: CLI arg > environment variable
    api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("No API key provided. Use --api-key or set OPENAI_API_KEY.")

    client = build_openai_client(
        api_base=config.api_base,
        api_key=api_key,
        max_concurrency=config.max_concurrency,
        request_timeout_s=config.request_timeout_s,
    )

    sem = asyncio.Semaphore(config.max_concurrency)
    completed_lock = asyncio.Lock()

    remaining = None if config.max_to_process <= 0 else config.max_to_process

    # Read report CSV in chunks to handle large datasets
    for df in pd.read_csv(
        config.reports_csv_path,
        chunksize=config.batch_size,
    ):
        # Use the configured report ID column
        report_id_col = config.report_id_column
        if report_id_col not in df.columns:
            raise KeyError(
                f"Report ID column '{report_id_col}' not found in CSV. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Ensure report IDs are strings for consistent comparison
        df[report_id_col] = df[report_id_col].astype(str).str.strip()

        # Filter already completed
        df = df[~df[report_id_col].isin(completed)]

        if df.empty:
            log.info("Batch: all rows already processed — skipping")
            continue

        # Identify metadata columns present in the dataframe
        metadata_columns = [
            c for c in config.metadata_columns
            if c in df.columns
        ]

        tasks = []
        for _, row in df.iterrows():
            if remaining is not None and remaining <= 0:
                break

            report_id = str(row[report_id_col])
            outfile = os.path.join(config.output_dir, f"{report_id}_{config.output_suffix}.json")
            if os.path.exists(outfile):
                continue

            tasks.append(process_one_row(
                sem, client, system_prompt, row, completed, completed_lock,
                config=config,
                json_schema=json_schema,
                validate_fn=validate_fn,
                report_text_column=config.report_text_column,
                exam_column=config.exam_column,
                report_id_column=report_id_col,
                output_suffix=config.output_suffix,
                user_message_template=config.user_message_template,
                metadata_columns=metadata_columns,
            ))
            scheduled_total += 1
            if remaining is not None:
                remaining -= 1

        if tasks:
            log.info("Batch: scheduling %d items (max_concurrency=%d)", len(tasks), config.max_concurrency)
            await asyncio.gather(*tasks)

        if config.max_to_process > 0 and scheduled_total >= config.max_to_process:
            return


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing with YAML config overlay
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the structuring/labeling pipeline.

    If --config is provided, loads the YAML file and uses its values as
    defaults for any argument not explicitly set on the command line.
    CLI arguments always take precedence over YAML config values.
    """
    ap = argparse.ArgumentParser(
        description="LLM-based radiology report structuring and labeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file (parsed first to set defaults) ──
    ap.add_argument("--config", type=str, default=None,
                    help="Path to a YAML config file. Keys use underscores "
                         "(e.g., report_text_column). CLI args override config values.")

    # ── Input / output paths ──
    ap.add_argument("--reports-csv-path", type=str, default=None,
                    help="Path to the CSV file containing free-text reports")
    ap.add_argument("--system-prompt-path", type=str, default=None,
                    help="Path to the Markdown file containing LLM system instructions")
    ap.add_argument("--schema-path", type=str, default=None,
                    help="Path to the JSON schema file defining the structured output format")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Directory to write structured JSON output files")
    ap.add_argument("--debug-dir", type=str, default=None,
                    help="Directory for debug dumps on failure (defaults to <output_dir>/debug)")
    ap.add_argument("--output-suffix", type=str, default="structured",
                    help="Suffix for output files: <report_id>_<suffix>.json")

    # ── CSV column configuration ──
    ap.add_argument("--report-id-column", type=str, default="report_id",
                    help="Column name in the CSV containing the unique report identifier")
    ap.add_argument("--report-text-column", type=str, default="report_text",
                    help="Column name in the CSV containing the report free-text")
    ap.add_argument("--exam-column", type=str, default="examination",
                    help="Column name for examination/procedure description (set empty to disable)")
    ap.add_argument("--metadata-columns", type=str, default="report_id",
                    help="Comma-separated column names to include in the output metadata")

    # ── User message template ──
    ap.add_argument("--user-message-template", type=str,
                    default="Input (free text):\nExamination: {exam_description}\nReport text: {report_text}\n",
                    help="Template for the user message sent to the LLM. "
                         "Available placeholders: {report_text}, {exam_description}")

    # ── API configuration ──
    ap.add_argument("--api-base", type=str, default="",
                    help="Base URL for the OpenAI-compatible API endpoint")
    ap.add_argument("--api-key", type=str, default="",
                    help="API key (can also be set via OPENAI_API_KEY env var)")
    ap.add_argument("--model-name", type=str, default=None,
                    help="Model name/identifier to use for completions")

    # ── Processing controls ──
    ap.add_argument("--batch-size", type=int, default=500000,
                    help="Number of CSV rows per chunk (for memory-efficient streaming)")
    ap.add_argument("--max-to-process", type=int, default=0,
                    help="Maximum number of reports to process (0 = no limit)")

    # ── Parallelism and pacing ──
    ap.add_argument("--max-concurrency", type=int, default=15,
                    help="Maximum number of concurrent LLM requests")

    # ── Timeouts and retries ──
    ap.add_argument("--request-timeout-s", type=float, default=60.0,
                    help="Per-request timeout in seconds")
    ap.add_argument("--max-retries", type=int, default=10,
                    help="Maximum number of retry attempts per report")
    ap.add_argument("--initial-backoff-s", type=float, default=1.0,
                    help="Initial backoff duration before retry")
    ap.add_argument("--max-backoff-s", type=float, default=60.0,
                    help="Maximum backoff cap")
    ap.add_argument("--max-jitter-s", type=float, default=0.75,
                    help="Maximum random jitter added to backoff sleep")
    ap.add_argument("--retryable-statuses", type=int, nargs="+",
                    default=[408, 429, 500, 502, 503, 504],
                    help="HTTP status codes that trigger automatic retry")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="LLM sampling temperature (0.0 = deterministic)")

    # ── Optional provider-specific parameters ──
    ap.add_argument("--service-tier", type=str, default=None,
                    help="Optional service tier parameter (provider-specific)")
    ap.add_argument("--reasoning-effort-default", type=str, default=None,
                    help="Default reasoning effort level (e.g., 'low', 'medium')")
    ap.add_argument("--reasoning-effort-on-retry", type=str, default=None,
                    help="Reasoning effort level on validation-error retries")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="Maximum number of tokens for the LLM response")

    # First pass: parse to check for --config
    args = ap.parse_args()

    # If a config file is provided, load it and apply as defaults for
    # any argument that was not explicitly set on the command line.
    if args.config:
        yaml_config = load_config(args.config)
        log.info("Loaded YAML config from %s (%d keys)", args.config, len(yaml_config))

        # Determine which args were explicitly set on CLI by comparing
        # against defaults. We re-parse with modified defaults.
        # Strategy: set_defaults from YAML, then re-parse so CLI wins.
        yaml_defaults = {}
        for key, value in yaml_config.items():
            # YAML keys use underscores; argparse dest also uses underscores
            yaml_defaults[key] = value

        ap.set_defaults(**yaml_defaults)
        args = ap.parse_args()

    # Validate required arguments (may come from config or CLI)
    missing = []
    for req in ("reports_csv_path", "system_prompt_path", "schema_path", "output_dir", "model_name"):
        if not getattr(args, req, None):
            missing.append(f"--{req.replace('_', '-')}")
    if missing:
        ap.error(f"The following arguments are required: {', '.join(missing)} "
                 f"(set via CLI or in the YAML config file)")

    return args


def main():
    """Entry point: parse args, create directories, and run the async pipeline."""
    config = parse_args()

    # Set defaults and create directories
    if not config.debug_dir:
        config.debug_dir = os.path.join(config.output_dir, "debug")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.debug_dir).mkdir(parents=True, exist_ok=True)

    # Parse metadata columns list
    config.metadata_columns = [c.strip() for c in config.metadata_columns.split(",") if c.strip()]

    # Handle empty exam column
    if config.exam_column == "":
        config.exam_column = None

    t0 = time.perf_counter()
    try:
        asyncio.run(main_async(config))
    finally:
        log.info("Done in %.2fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
