"""
13_portfolio_cea.py — What the current portfolio of recommendations would avert,
and what the same money would avert allocated by evidence.

Implementation intensity. Committees do not attach budgets to recommendations,
so the model has to translate attention into effort. Let s_L be the share of
recommendations pulling lever L. Under the observed portfolio each intervention
in lever L is implemented at a coverage of s_L divided by the largest lever
share, multiplied by that intervention's maximum attainable coverage. The
most-recommended lever is therefore implemented as fully as it can be and every
other lever scales down in proportion to how often committees ask for it. That
fixes a cost, and every comparator is then required to cost the same.

Four portfolios are compared at that identical cost.
  observed              coverage set by recommendation shares, as above
  within-lever optimal  the same dollars per lever, allocated well inside each
  evidence-weighted     all dollars allocated to maximize deaths averted
  burden-aligned        dollars split across causes by national cause-specific
                        mortality share, then allocated well within each cause

The gap between observed and within-lever optimal is misallocation inside the
levers committees already favor. The gap between within-lever optimal and
evidence-weighted is misallocation across levers. Reporting both separates a
problem of execution from a problem of attention.

Output: results/portfolio_cea.json
"""
from __future__ import annotations
import copy, importlib.util, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, HERE / path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


ms = _load("ms", "11_microsim.py")
iv = _load("iv", "12_interventions.py")

# Share of a postpartum intervention's effect that is reachable when coverage
# ends at 60 days. Derived from the national distribution of the timing of
# pregnancy-related deaths: deaths through 42 days as a share of all deaths
# after delivery.
def postpartum_reachable(P: dict) -> float:
    t = P["timing_of_pregnancy_related_deaths"]["value"]
    early = t["day_of_delivery_through_6_days_postpartum"] + t["7_to_42_days_postpartum"]
    late = t["43_to_365_days_postpartum"]
    return early / (early + late)


def life_expectancy(age: np.ndarray, le30: float) -> np.ndarray:
    """Remaining life expectancy, linear in age across the reproductive years.

    The female life table is close to linear between 15 and 45, losing a little
    under one year of remaining expectancy per year of age.
    """
    return np.clip(le30 + (30.0 - age) * 0.93, 1.0, 75.0)


def annuity(years: np.ndarray, r: float) -> np.ndarray:
    return years if r == 0 else (1 - (1 + r) ** (-years)) / r


class Model:
    def __init__(self, P: dict, n: int, seed: int, all_extended: bool | None = None,
                 current: dict | None = None):
        """current, if given, is coverage already in place before any portfolio.
        The baseline is then evaluated with it, portfolios add coverage on top of
        it up to each intervention's maximum reach, and only the added coverage is
        costed. Without it every portfolio is compared against no implementation."""
        self.P = P
        self.current = current or {}
        self.rng = np.random.default_rng(seed)
        self.c = ms.build_cohort(P, n, self.rng, all_extended=all_extended)
        self.cal = ms.calibrate(self.c, P)
        self.ints = {i.key: i for i in iv.build(P)}
        self.pp_reach = postpartum_reachable(P)
        self.elig = {k: i.eligible(self.c) for k, i in self.ints.items()}
        self.n_elig = {k: float(v.sum()) * self.c.weight for k, v in self.elig.items()}
        self.base = ms.evaluate(self.c, self.cal, self.multipliers({}, include_current=True))
        for k, v in self.current.items():
            if k in self.ints:
                self.ints[k].max_reach = max(self.ints[k].max_reach - v, 0.0)
        u = P["utility_healthy_reproductive_age_female"]["value"]
        r = P["discount_rate"]["value"]
        self.qaly_per_death = u * annuity(
            life_expectancy(self.c.age, P["life_expectancy_female_30"]["value"]), r)

    # ---- mechanics ----------------------------------------------------------
    def multipliers(self, cov: dict, include_current: bool = True) -> dict:
        """Per-cause, per-stage multiplier arrays for a coverage vector, with any
        coverage already in place added on top."""
        mult: dict = {cz: {} for cz in ms.CAUSES}
        for k, i in self.ints.items():
            c_j = cov.get(k, 0.0) + (self.current.get(k, 0.0) if include_current else 0.0)
            if c_j <= 0 or i.stage == "none" or i.stage == "exposure" or not i.causes:
                continue
            eff = self.elig[k].astype(float) * c_j
            if i.postpartum:
                # Outside a twelve-month coverage state only the part of the
                # postpartum window that is covered can be reached.
                eff = eff * np.where(self.c.extension, 1.0, self.pp_reach)
            m = 1.0 - eff * (1.0 - i.rr)
            for cz in i.causes:
                mult[cz][i.stage] = mult[cz].get(i.stage, 1.0) * m
        return mult

    def cost(self, cov: dict) -> float:
        return sum(self.ints[k].unit_cost * self.n_elig[k] * c for k, c in cov.items() if c > 0)

    def outcomes(self, cov: dict) -> dict:
        r = ms.evaluate(self.c, self.cal, self.multipliers(cov))
        b, t = self.base["totals"], r["totals"]
        deaths_averted = b["deaths"] - t["deaths"]
        smm_averted = b["smm_events"] - t["smm_events"]
        dep_averted = (b["complications"]["Mental health conditions"]
                       - t["complications"]["Mental health conditions"])

        # Quality-adjusted life years. Deaths averted carry discounted remaining
        # life at the individual's own age. Severe morbidity carries a first-year
        # decrement that is zero in the base case because no published estimate
        # exists. Perinatal depression averted carries the published decrement
        # for half a year.
        dq = np.zeros(self.c.n)
        for cz in ms.CAUSES:
            dq += (self.base["death"][cz] - r["death"][cz]) * self.qaly_per_death
        qaly = float(dq.sum() * self.c.weight)
        smm_dec = self.P["utility_decrement_smm_survivor"]["value"]
        qaly += smm_averted * smm_dec
        qaly += dep_averted * self.P["utility_decrement_postpartum_depression"]["value"] * 0.5

        prog = self.cost(cov)
        offset = smm_averted * self.P["cost_smm_increment"]["value"]
        return {"coverage": {k: round(v, 4) for k, v in cov.items() if v > 0},
                "program_cost": prog, "cost_offset_smm": offset,
                "net_cost": prog - offset,
                "deaths": t["deaths"], "deaths_averted": deaths_averted,
                "smm_events": t["smm_events"], "smm_averted": smm_averted,
                "depression_averted": dep_averted,
                "qalys_gained": qaly,
                "cost_per_qaly": (prog - offset) / qaly if qaly > 0 else float("inf"),
                "cost_per_death_averted": (prog - offset) / deaths_averted
                if deaths_averted > 0 else float("inf"),
                "deaths_by_cause": t["deaths_by_cause"]}

    # ---- portfolios ----------------------------------------------------------
    def observed(self, lever_share: dict, intensity: float = 1.0) -> dict:
        top = max(lever_share.values())
        return {k: min(i.max_reach,
                       intensity * lever_share.get(i.lever, 0.0) / top * i.max_reach)
                for k, i in self.ints.items()}

    def observed_from_interventions(self, int_share: dict, intensity: float = 1.0) -> dict:
        """Coverage set by how often committees ask for each specific intervention.

        The most-requested intervention in the registry is implemented at its
        maximum attainable reach and every other scales down in proportion to how
        often it is requested. Recommendations that name no registry intervention
        carry no cost and no effect here; they are counted in the classification
        results and priced in the lever-based sensitivity analysis instead.
        """
        top = max(int_share.values())
        return {k: min(i.max_reach, intensity * int_share.get(k, 0.0) / top * i.max_reach)
                for k, i in self.ints.items()}

    def requested_full(self, int_share: dict, exclude: tuple = ()) -> dict:
        """Primary requested portfolio: every intervention requested by at least
        one recommendation, implemented at its maximum attainable reach. This is
        the most generous reading of the recommendations, and it does not depend
        on treating request counts as budget weights."""
        return {k: (i.max_reach if int_share.get(k, 0) > 0 and k not in exclude else 0.0)
                for k, i in self.ints.items()}

    def requested_frequency(self, int_share: dict, effective_norm: bool = False) -> dict:
        """Coverage proportional to request frequency, scaled so that the most
        requested intervention, or the most requested intervention with an
        effect, is at its maximum reach."""
        pool = {k: v for k, v in int_share.items()
                if not effective_norm or (k in self.ints and self.ints[k].rr < 1.0
                                          and self.ints[k].stage in ("incidence", "smm", "cfr"))}
        top = max(pool.values())
        return {k: min(i.max_reach, int_share.get(k, 0.0) / top * i.max_reach)
                for k, i in self.ints.items()}

    def burden_aligned_at_budget(self, budget: float) -> dict:
        """Burden-aligned coverage with its intensity scaled so that its cost does
        not exceed the reference budget."""
        lo, hi = 0.0, 1.0
        if self.cost(self.burden_aligned(1.0)) <= budget:
            return self.burden_aligned(1.0)
        for _ in range(40):
            mid = (lo + hi) / 2
            if self.cost(self.burden_aligned(mid)) > budget:
                hi = mid
            else:
                lo = mid
        return self.burden_aligned(lo)

    def greedy(self, budget: float, keys: list[str] | None = None,
               start: dict | None = None, steps: int = 300) -> dict:
        """Allocate a budget to maximize deaths averted, in equal increments."""
        keys = keys or list(self.ints)
        cov = dict(start or {k: 0.0 for k in self.ints})
        spent = self.cost(cov)
        if spent > budget:
            return cov
        inc = (budget - spent) / steps
        base_deaths = self.outcomes(cov)["deaths"]
        for _ in range(steps):
            best, best_gain = None, 0.0
            for k in keys:
                i = self.ints[k]
                if cov[k] >= i.max_reach - 1e-9 or i.stage in ("none", "exposure"):
                    continue
                unit = i.unit_cost * self.n_elig[k]
                if unit <= 0:
                    continue
                d = min(inc / unit, i.max_reach - cov[k])
                trial = dict(cov); trial[k] = cov[k] + d
                gain = base_deaths - self.outcomes(trial)["deaths"]
                per_dollar = gain / (d * unit) if d * unit > 0 else 0.0
                if per_dollar > best_gain:
                    best, best_gain, best_d = k, per_dollar, d
            if best is None:
                break
            cov[best] += best_d
            base_deaths = self.outcomes(cov)["deaths"]
        return cov

    def within_lever(self, budget_by_lever: dict, steps: int = 60) -> dict:
        cov = {k: 0.0 for k in self.ints}
        for lever, b in budget_by_lever.items():
            keys = [k for k, i in self.ints.items() if i.lever == lever]
            if not keys or b <= 0:
                continue
            sub = self.greedy(b, keys=keys, start={k: 0.0 for k in self.ints}, steps=steps)
            for k in keys:
                cov[k] = sub[k]
        return cov

    def burden_aligned(self, intensity: float = 1.0) -> dict:
        """The same kinds of intervention, distributed by burden rather than by
        recommendation counts.

        Each intervention's coverage is set by the share of pregnancy-related
        deaths its cause accounts for, normalized so the largest cause is
        implemented as fully as it can be. Interventions that address no single
        cause take the mean share, since that is what an even-handed committee
        would give something cross-cutting.
        """
        sh = self.P["cause_distribution_prmr"]["value"]
        top = max(sh.values())
        mean_sh = sum(sh.values()) / len(sh)
        cov = {}
        for k, i in self.ints.items():
            w = (max(sh[cz] for cz in i.causes) if i.causes else mean_sh) / top
            cov[k] = min(i.max_reach, intensity * w * i.max_reach)
        return cov

    def full_effective_menu(self) -> dict:
        """Every intervention with a non-null effect, at its maximum reach."""
        return {k: (i.max_reach if i.rr < 1.0 and i.stage in ("incidence", "smm", "cfr") else 0.0)
                for k, i in self.ints.items()}

    def frontier(self, budget_max: float, points: int = 24) -> list[dict]:
        """Deaths averted at the best allocation of each of a series of budgets.

        Budgets are spaced logarithmically because the effective interventions
        are cheap and the curve is nearly flat above a few hundred million.
        """
        out = []
        for b in np.geomspace(budget_max / 2000.0, budget_max, points):
            r = self.outcomes(self.greedy(float(b), steps=90))
            out.append({"budget": float(b), "spent": r["program_cost"],
                        "deaths_averted": r["deaths_averted"],
                        "qalys_gained": r["qalys_gained"]})
        return out

    def greedy_path(self, steps: int = 200) -> list[tuple[float, float]]:
        """Cumulative spend and deaths averted along the greedy allocation, run to
        the budget at which every effective intervention is at full reach. The
        greedy order is the order of deaths averted per dollar, so the path is the
        efficiency frontier."""
        cap = self.cost(self.full_effective_menu())
        cov = {k: 0.0 for k in self.ints}
        inc = cap / steps
        base = self.outcomes(cov)["deaths"]
        d0 = self.base["totals"]["deaths"]
        path = [(0.0, 0.0)]
        for _ in range(steps * 2):
            best, best_gain, best_d = None, 0.0, 0.0
            for k, i in self.ints.items():
                if cov[k] >= i.max_reach - 1e-9 or i.stage in ("none", "exposure"):
                    continue
                unit = i.unit_cost * self.n_elig[k]
                if unit <= 0:
                    continue
                d = min(inc / unit, i.max_reach - cov[k])
                trial = dict(cov); trial[k] = cov[k] + d
                g = (base - self.outcomes(trial)["deaths"]) / (d * unit)
                if g > best_gain:
                    best, best_gain, best_d = k, g, d
            if best is None:
                break
            cov[best] += best_d
            base = self.outcomes(cov)["deaths"]
            path.append((self.cost(cov), d0 - base))
        return path

    def equivalent_budget(self, target_deaths: float, budget_max: float) -> dict:
        """The smallest spend at which the efficiency frontier matches a given
        number of deaths averted, by interpolation along the greedy path."""
        path = self.greedy_path()
        for (c0, y0), (c1, y1) in zip(path, path[1:]):
            if y1 >= target_deaths:
                c = c0 + (c1 - c0) * ((target_deaths - y0) / (y1 - y0) if y1 > y0 else 0)
                return {"budget": c, "deaths_averted": target_deaths,
                        "share_of_observed_budget": c / budget_max,
                        "frontier_max_deaths": path[-1][1], "frontier_max_spend": path[-1][0]}
        return {"budget": None, "share_of_observed_budget": None,
                "note": "target exceeds the frontier maximum",
                "frontier_max_deaths": path[-1][1], "frontier_max_spend": path[-1][0]}


def intervention_shares_from(path: Path) -> dict:
    """Share of all recommendations requesting each registry intervention."""
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    c: dict = {}
    for r in rows:
        if r["intervention"] != "none":
            c[r["intervention"]] = c.get(r["intervention"], 0) + 1
    return {k: v / n for k, v in c.items()}


def lever_shares_from(path: Path, which: str = "pass_A") -> dict:
    d = json.load(open(path))
    return {r["label"]: r["pct"] / 100.0 for r in d[which]["policy_lever"]}


def main() -> None:
    P = ms.load_params()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    m = Model(P, n, seed=20260924)
    shares = lever_shares_from(RES / "classification.json")
    ishares = intervention_shares_from(RES / "intervention_map_passA.jsonl")

    # Primary specification: every requested intervention at full reach.
    cov_obs = m.requested_full(ishares)
    obs = m.outcomes(cov_obs)
    budget = obs["program_cost"]

    lever_budget, lever_cov = {}, {}
    for k, c in cov_obs.items():
        i = m.ints[k]
        lever_budget[i.lever] = lever_budget.get(i.lever, 0.0) + i.unit_cost * m.n_elig[k] * c
        lever_cov.setdefault(i.lever, {})[k] = round(c, 4)

    res = {
        "observed": obs,
        "evidence_weighted": m.outcomes(m.greedy(budget)),
        "within_lever_optimal": m.outcomes(m.within_lever(lever_budget)),
        "burden_aligned": m.outcomes(m.burden_aligned_at_budget(budget)),
        "burden_aligned_unconstrained": m.outcomes(m.burden_aligned()),
        "full_effective_menu": m.outcomes(m.full_effective_menu()),
        "requested_frequency": m.outcomes(m.requested_frequency(ishares)),
        "requested_frequency_effective_norm": m.outcomes(m.requested_frequency(ishares, True)),
        "requested_excluding_home_visiting": m.outcomes(
            m.requested_full(ishares, exclude=("nurse_home_visiting",))),
        "observed_lever_based": m.outcomes(m.observed(shares)),
    }
    # Incremental over current practice
    cur = json.load(open(RES / "current_practice.json"))["coverage"]
    mc = Model(P, n, seed=20260924, current=cur)
    oc = mc.outcomes(mc.requested_full(ishares))
    res["observed_incremental_current_practice"] = oc
    res["evidence_weighted_incremental_current_practice"] = mc.outcomes(mc.greedy(oc["program_cost"]))
    front = m.frontier(budget)
    equiv = m.equivalent_budget(obs["deaths_averted"], budget)

    b = m.base["totals"]
    print(f"cohort {n:,} simulated; {b['births']:,.0f} Medicaid-financed births")
    print(f"baseline: {b['deaths']:,.0f} pregnancy-related deaths, "
          f"{b['smm_events']:,.0f} severe maternal morbidity events\n")
    print(f"{'portfolio':24s} {'deaths averted':>14s} {'SMM averted':>12s} {'QALYs':>8s} "
          f"{'program $M':>11s} {'net $M':>9s} {'$/QALY':>11s}")
    for k, r in res.items():
        cpq = r["cost_per_qaly"]
        print(f"{k:24s} {r['deaths_averted']:14,.0f} {r['smm_averted']:12,.0f} "
              f"{r['qalys_gained']:8,.0f} {r['program_cost']/1e6:11,.0f} "
              f"{r['net_cost']/1e6:9,.0f} "
              f"{cpq:11,.0f}" if cpq < 1e8 else f"{k:24s} ... dominated")

    print(f"\nobserved portfolio cost by policy lever")
    for lv, v in sorted(lever_budget.items(), key=lambda kv: -kv[1]):
        print(f"  {lv:34s} ${v/1e6:9,.0f}M  ({100*v/budget:5.1f}% of spend, "
              f"{100*shares.get(lv,0):4.1f}% of recommendations)")

    ev, ob = res["evidence_weighted"], res["observed"]
    if equiv.get("budget"):
        print(f"\na well-allocated portfolio matches the observed portfolio's "
              f"{obs['deaths_averted']:.0f} deaths averted at a budget of "
              f"${equiv['budget']/1e6:,.0f}M, which is "
              f"{100*equiv['share_of_observed_budget']:.1f}% of the observed portfolio's cost")
    print(f"\nthe evidence-weighted allocation averts "
          f"{ev['deaths_averted']/max(ob['deaths_averted'],1e-9):.1f} times as many deaths "
          f"for {100*ev['program_cost']/ob['program_cost']:.0f}% of the cost")

    json.dump({"n_simulated": n, "seed": 20260924, "baseline": b,
               "observed_budget": budget, "budget_per_birth": budget / b["births"],
               "lever_shares": shares, "intervention_shares": ishares,
               "lever_budget": lever_budget,
               "lever_coverage": lever_cov, "portfolios": res, "frontier": front,
               "equivalent_budget": equiv,
               "postpartum_reachable_without_extension": m.pp_reach,
               "calibration": {"kappa": m.cal["kappa"],
                               "achieved_smm_per_10k": m.cal["achieved_smm_per_10k"],
                               "case_fatality_check": m.cal["case_fatality_check"],
                               "progression_clipped_at_one": m.cal["clipped_at_one"],
                               "cfr": m.cal["cfr"],
                               "p_smm_given_comp": m.cal["p_smm_given_comp"]}},
              open(RES / "portfolio_cea.json", "w"), indent=1)
    print(f"\nwrote {RES/'portfolio_cea.json'}")


if __name__ == "__main__":
    main()
