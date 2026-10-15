"""Process-safe cumulative reservations for the user-authorized pilot ceiling."""

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import time


class PilotBudget:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def transaction(self):
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = json.loads(self.path.read_text())
            yield ledger
            ledger["updated_at_unix"] = time.time()
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(ledger, indent=2) + "\n")
            os.replace(tmp, self.path)

    @staticmethod
    def committed(ledger):
        return sum(item["amount_usd"] for item in ledger["entries"].values())

    def reserve(self, key, amount, kind):
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("positive finite reservation required")
        with self.transaction() as ledger:
            if key in ledger["entries"]:
                raise ValueError(
                    "run already reserved; never silently repeat paid work"
                )
            if self.committed(ledger) + amount > ledger["cap_usd"]:
                raise RuntimeError(
                    "cumulative pilot budget cannot cover this reservation"
                )
            ledger["entries"][key] = {
                "kind": kind,
                "status": "reserved",
                "amount_usd": amount,
                "original_reservation_usd": amount,
                "created_at_unix": time.time(),
            }

    def settle(self, key, amount, note):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("nonnegative finite settlement required")
        overrun = False
        with self.transaction() as ledger:
            entry = ledger["entries"][key]
            if entry["status"] != "reserved":
                raise ValueError("reservation already settled")
            overrun = amount > entry["original_reservation_usd"] + 1e-9
            entry.update(
                status="settled",
                amount_usd=amount,
                note=note,
                settled_at_unix=time.time(),
                reservation_exceeded=overrun,
            )
        if overrun:
            raise RuntimeError(
                "provider exceeded reservation; actual liability recorded"
            )
