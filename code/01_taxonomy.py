"""
01_taxonomy.py — Harmonize free-text MMRC cause-of-death strings and
recommendation categories onto the CDC MMRIA / ERASE MM cause taxonomy.

The corpus carries 422 distinct `leading_cause` strings and 2,087 distinct
recommendation `category` strings because every state writes its own labels.
Both must land on one vocabulary before burden and recommendation content can
be compared within a state.

Ordering matters: the rules below are applied in sequence and the first match
wins, so more specific patterns (cardiomyopathy, amniotic fluid embolism) are
tested before the general ones (cardiovascular, embolism). Mental health is
tested before substance use only where a state's label explicitly combines
them, since CDC counts substance-related deaths under mental health conditions
but many states report them separately; we keep them separate and provide a
combined view for sensitivity.

Exported:
    DOMAINS         canonical domain list
    map_cause(s)    cause string  -> domain or None
    map_recommendation(text, category) -> domain or None
"""
from __future__ import annotations
import re

DOMAINS = [
    "Mental health conditions",
    "Substance use disorder",
    "Cardiovascular conditions",
    "Cardiomyopathy",
    "Hemorrhage",
    "Infection",
    "Embolism",
    "Hypertensive disorders",
    "Cerebrovascular accident",
    "Injury, homicide, and violence",
    "Other medical conditions",
]

# (domain, pattern) applied in order; first match wins.
_CAUSE_RULES: list[tuple[str, str]] = [
    ("Cardiomyopathy", r"cardiomyopath|peripartum cardiomy"),
    ("Embolism", r"amniotic fluid embol|thrombotic embol|pulmonary embol|\bembol"),
    ("Substance use disorder", r"substance use|overdose|\bpoison|opioid|drug[- ]related|\bsud\b"),
    ("Mental health conditions", r"mental health|suicid|self[- ]harm|psychiatric|depress"),
    ("Hemorrhage", r"h[ae]morrhag|obstetric bleeding|placenta accreta|uterine atony|hysterectomy"),
    ("Infection", r"infection|sepsis|septic|covid|influenza|ards|acute respiratory distress|pneumon"),
    ("Hypertensive disorders", r"hypertens|preeclamp|pre[- ]eclamp|eclampsia|hellp"),
    ("Cerebrovascular accident", r"cerebrovascular|stroke|intracranial|subarachnoid"),
    ("Injury, homicide, and violence", r"homicide|firearm|violence|injury|motor vehicle|accident|trauma|assault"),
    ("Cardiovascular conditions", r"cardiovascular|cardiac|coronary|heart"),
    ("Other medical conditions", r"cancer|malignan|autoimmune|endocrine|renal|hepat|anesthesia|medical condition"),
]

# Recommendation text is longer and noisier than a cause label, so these rules
# key on intervention language as well as condition language.
_REC_RULES: list[tuple[str, str]] = [
    ("Substance use disorder", r"substance use|\bsud\b|opioid|overdose|naloxone|buprenorphine|methadone|harm reduction|\bmoud\b|\bmat\b|addiction"),
    ("Mental health conditions", r"mental health|behavioral health|depress|suicid|psychiatr|perinatal mood|\bppd\b|anxiety|screening for depression|edinburgh|\bphq"),
    ("Cardiomyopathy", r"cardiomyopath"),
    ("Hypertensive disorders", r"hypertens|preeclamp|pre[- ]eclamp|eclampsia|hellp|blood pressure|\bbp\b"),
    ("Hemorrhage", r"h[ae]morrhag|blood loss|quantitative blood|massive transfusion|tranexamic|uterotonic|oxytocin"),
    ("Infection", r"infection|sepsis|septic|covid|vaccinat|immuniz"),
    ("Embolism", r"embol|thromboprophylax|venous thromboembol|\bvte\b|anticoagul"),
    ("Cerebrovascular accident", r"cerebrovascular|stroke"),
    ("Injury, homicide, and violence", r"homicide|firearm|intimate partner violence|\bipv\b|domestic violence|violence|injury|motor vehicle|human traffick"),
    ("Cardiovascular conditions", r"cardiovascular|cardiac|coronary|\bheart\b"),
    ("Other medical conditions", r"diabet|obesity|cancer|anesthes|renal|asthma|chronic disease"),
]


def _first_match(text: str, rules: list[tuple[str, str]]) -> str | None:
    t = (text or "").lower()
    if not t.strip():
        return None
    for domain, pattern in rules:
        if re.search(pattern, t):
            return domain
    return None


def map_cause(s: str | None) -> str | None:
    """Map a state's free-text leading-cause label to a canonical domain."""
    return _first_match(s or "", _CAUSE_RULES)


def map_recommendation(recommendation: str | None, category: str | None = None) -> str | None:
    """Map a recommendation to the cause domain it addresses.

    The category field is checked first because states use it as a topic header;
    the recommendation body is the fallback.
    """
    return _first_match(category or "", _REC_RULES) or _first_match(recommendation or "", _REC_RULES)


# Intervention lever, orthogonal to cause domain: what is actually being asked for.
_LEVER_RULES: list[tuple[str, str]] = [
    ("Coverage or payment", r"\bcoverage\b|reimburs|\bbenefit\b|\bfund\b|funding|\bpay for\b|payment|medicaid expansion|extend.{0,30}postpartum"),
    ("Workforce or staffing", r"\bworkforce\b|staffing|recruit|midwif|doula|community health worker|\bchw\b|home visit"),
    ("Clinical protocol or bundle", r"\bbundle\b|protocol|guideline|checklist|standard of care|simulation drill|\bdrill\b|levels of maternal care|risk[- ]appropriate"),
    ("Screening or referral", r"\bscreen|\breferral|assess for|universal screening"),
    ("Data, surveillance, or review", r"\bdata\b|surveillance|registry|review committee|reporting|linkage|dashboard"),
    ("Education or training", r"\beducat|\btrain|\bawareness|curricul|continuing education|\bcme\b"),
    ("Care coordination or navigation", r"care coordination|case management|navigat|continuity of care|transition of care|follow[- ]up"),
    ("Community or social services", r"housing|transport|food|social determinant|community[- ]based|social service|child care"),
]


def map_lever(recommendation: str | None, category: str | None = None) -> str | None:
    """Map a recommendation to the policy lever it pulls."""
    return _first_match(recommendation or "", _LEVER_RULES) or _first_match(category or "", _LEVER_RULES)


LEVERS = [d for d, _ in _LEVER_RULES]
