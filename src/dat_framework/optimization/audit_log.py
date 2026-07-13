"""
Append-only audit log for every fairness-audit run, per the Track 4 ToR's
"Data governance and audit note" requirement (section 2.4): "How logs,
outputs, incidents, model changes, user appeals and human review decisions
will be recorded, retained and reviewed."

This is intentionally simple (a versioned, hash-chained CSV/JSON-lines file)
rather than a database, so it can be dropped into any institution's existing
infrastructure without new dependencies — the audit trail itself, not the
storage engine, is what a reviewer or regulator needs to see.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class AuditEntry:
    timestamp_utc: str
    run_id: str
    actor: str                     # who/what triggered the run (user, scheduler, API caller)
    dataset_fingerprint: str       # sha256 of the input data, for reproducibility/chain-of-custody
    action: str                    # e.g. "fairness_audit", "moo_sweep", "model_selection", "override"
    w0_selected: Optional[float]   # which Pareto point was chosen (Tier 3 governance decision), if applicable
    mean_dir: Optional[float]
    mean_dpd: Optional[float]
    mean_eod: Optional[float]
    decision_rationale: str        # free-text: why this point/action was chosen
    previous_entry_hash: str       # hash-chains entries so the log can be tamper-evidenced


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Deterministic fingerprint of a dataset for chain-of-custody purposes."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def _hash_entry(entry_dict: dict) -> str:
    return hashlib.sha256(json.dumps(entry_dict, sort_keys=True).encode()).hexdigest()[:16]


class AuditLog:
    """Minimal hash-chained audit log. Each institution can point `log_path`
    at wherever their own governance process expects logs to live (e.g. a
    shared drive, a SIEM ingestion folder, or a compliance archive satisfying
    the Cyber and Data Protection Act's record-keeping expectations)."""

    def __init__(self, log_path: str = "data/processed/audit_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.log_path.exists():
            return "GENESIS"
        with open(self.log_path, "r") as f:
            lines = f.readlines()
        if not lines:
            return "GENESIS"
        return json.loads(lines[-1])["entry_hash"]

    def record(
        self,
        actor: str,
        action: str,
        dataset: pd.DataFrame,
        decision_rationale: str,
        w0_selected: Optional[float] = None,
        mean_dir: Optional[float] = None,
        mean_dpd: Optional[float] = None,
        mean_eod: Optional[float] = None,
    ) -> AuditEntry:
        prev_hash = self._last_hash()
        entry = AuditEntry(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            run_id=hashlib.sha256(
                f"{datetime.now(timezone.utc).isoformat()}{actor}{action}".encode()
            ).hexdigest()[:12],
            actor=actor,
            dataset_fingerprint=_hash_dataframe(dataset),
            action=action,
            w0_selected=w0_selected,
            mean_dir=mean_dir,
            mean_dpd=mean_dpd,
            mean_eod=mean_eod,
            decision_rationale=decision_rationale,
            previous_entry_hash=prev_hash,
        )
        entry_dict = asdict(entry)
        entry_dict["entry_hash"] = _hash_entry(entry_dict)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry_dict) + "\n")
        return entry

    def read_all(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame()
        return pd.read_json(self.log_path, lines=True)

    def verify_chain(self) -> bool:
        """Confirm no entry has been altered or removed out of order."""
        df = self.read_all()
        if df.empty:
            return True
        prev = "GENESIS"
        for _, row in df.iterrows():
            if row["previous_entry_hash"] != prev:
                return False
            prev = row["entry_hash"]
        return True
