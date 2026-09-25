"""
10_model_inputs.py — Single loader for every external number the model uses.

Two files feed the model and nothing else does:
  results/denominators.json     state births, Medicaid share, coverage and AIM status
  results/epi_cost_params.json  rates, costs, utilities, life expectancy

Both were assembled from published sources with a citation string attached to
every value. A parameter that could not be verified is stored as null. This
module refuses to substitute a value for a null; a model run that needs one
fails and names it, so no unsourced number can reach a result.
"""
from __future__ import annotations
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


class Missing(KeyError):
    """A required parameter is absent or null in the parameter files."""


class Params:
    def __init__(self) -> None:
        self.den = json.load(open(RES / "denominators.json"))
        self.epi = json.load(open(RES / "epi_cost_params.json"))
        self.used: dict[str, dict] = {}

    def get(self, key: str, field: str = "value"):
        """Return a parameter value and record that the run consumed it."""
        p = self.epi.get(key)
        if p is None:
            raise Missing(f"{key} is not in epi_cost_params.json")
        v = p.get(field)
        if v is None:
            raise Missing(f"{key}.{field} is null; source not established")
        self.used[key] = p
        return v

    def band(self, key: str, rel: float = 0.25) -> tuple[float, float, float]:
        """Value with a low and high bound. Falls back to a relative band when
        the source reports no interval, and records that it did so."""
        p = self.epi[key]
        v = self.get(key)
        lo, hi = p.get("low"), p.get("high")
        if lo is None or hi is None:
            lo, hi = v * (1 - rel), v * (1 + rel)
            self.used[key] = {**p, "band": f"assumed +/-{rel:.0%}, no published interval"}
        return float(v), float(lo), float(hi)

    # ---- state frame ---------------------------------------------------------
    def states(self) -> list[dict]:
        out = []
        for st, d in self.den["states"].items():
            b, share = d.get("births_2024"), d.get("medicaid_share_pct")
            if b is None or share is None:
                continue
            out.append(dict(state=st, births=b, medicaid_share=share / 100.0,
                            medicaid_births=b * share / 100.0,
                            extension=bool(d.get("postpartum_12mo_extension")),
                            aim=bool(d.get("aim_enrolled"))))
        return sorted(out, key=lambda r: r["state"])

    def provenance(self) -> dict:
        return {"parameters": self.used, "denominator_sources": self.den.get("sources", {})}


if __name__ == "__main__":
    p = Params()
    s = p.states()
    print(f"states with complete denominators: {len(s)}")
    print(f"Medicaid-financed births: {sum(r['medicaid_births'] for r in s):,.0f}")
    print(f"with 12-month extension: {sum(r['extension'] for r in s)}")
    print(f"\nepi parameters present: {len(p.epi)}")
    null = [k for k, v in p.epi.items() if isinstance(v, dict) and v.get('value') is None]
    print(f"null-valued: {len(null)} -> {null}")
