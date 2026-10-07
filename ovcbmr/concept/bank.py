"""Concept bank: the active vocabulary V (known concepts) + the held-out unseen
(Task-2 / unanchored) concepts, with prompt ensembles and absence templates.

A built-in DEFAULT_BANK keeps the package importable/testable without the YAML file;
`data/concepts/cxrlt2026_concepts.yaml` is the human-editable source of truth and takes
precedence when `concepts.bank` points at it.

NOTE: the exact CXR-LT 2026 class list comes from the challenge release — the names below
are a representative placeholder set. ADJUST `known`/`unseen` to the official 30 + 6.
"""
from __future__ import annotations

# Positive prompt templates (protocol §5: >=5 prompts/concept, averaged in text space).
PROMPT_TEMPLATES = (
    "chest x-ray showing {c}.",
    "radiographic evidence of {c}.",
    "findings consistent with {c}.",
    "there is {c}.",
    "{c} seen on the frontal chest radiograph.",
)
# Absence templates (presence-vs-absence separation).
NEGATIVE_TEMPLATES = (
    "no {c}.",
    "no evidence of {c}.",
    "clear chest radiograph without {c}.",
)


def _c(name, display, umls=""):
    return {"name": name, "display": display, "umls": umls}


# 30 known (Task 1) — the active vocabulary V. Placeholder, mirrors CXR-LT long-tail style.
_KNOWN = [
    _c("Cardiomegaly", "cardiomegaly"), _c("Pleural Effusion", "a pleural effusion"),
    _c("Consolidation", "consolidation"), _c("Atelectasis", "atelectasis"),
    _c("Pulmonary Edema", "pulmonary edema"), _c("Pneumonia", "pneumonia"),
    _c("Pneumothorax", "a pneumothorax"), _c("Emphysema", "emphysema"),
    _c("Fibrosis", "pulmonary fibrosis"), _c("Nodule", "a pulmonary nodule"),
    _c("Mass", "a pulmonary mass"), _c("Infiltration", "an infiltrate"),
    _c("Fracture", "a rib fracture"), _c("Hernia", "a hiatal hernia"),
    _c("Pleural Thickening", "pleural thickening"), _c("Support Devices", "a support device"),
    _c("Calcification", "a calcification"), _c("Tortuous Aorta", "a tortuous aorta"),
    _c("Cardiomediastinal", "an enlarged cardiomediastinal silhouette"),
    _c("Lung Opacity", "a lung opacity"), _c("Bronchiectasis", "bronchiectasis"),
    _c("Interstitial", "an interstitial pattern"), _c("Granuloma", "a granuloma"),
    _c("Aortic Elongation", "aortic elongation"), _c("Hilar Enlargement", "hilar enlargement"),
    _c("Costophrenic Angle", "costophrenic angle blunting"),
    _c("Scoliosis", "scoliosis"), _c("Air Trapping", "air trapping"),
    _c("Volume Loss", "volume loss"), _c("Normal", "no finding"),
]

# 6 unseen (Task 2) — held OUT of V at train; define the unanchored evaluation set.
_UNSEEN = [
    _c("Pneumoperitoneum", "pneumoperitoneum"),
    _c("Pneumomediastinum", "pneumomediastinum"),
    _c("Subcutaneous Emphysema", "subcutaneous emphysema"),
    _c("Pericardial Effusion", "a pericardial effusion"),
    _c("Chilaiditi Sign", "the Chilaiditi sign"),
    _c("Situs Inversus", "situs inversus"),
]

DEFAULT_BANK = {"known": _KNOWN, "unseen": _UNSEEN}


def load_bank(path=None):
    """Return the concept bank dict. path=None/'default' -> built-in; else load YAML."""
    if path in (None, "default", "built-in"):
        return DEFAULT_BANK
    import yaml  # lazy
    with open(path) as f:
        return yaml.safe_load(f)


def concept_names(bank, group="known"):
    return [c["name"] for c in bank[group]]


def active_vocabulary(bank):
    """The known-concept names that form V at train/inference."""
    return concept_names(bank, "known")


def _display(bank, name):
    for group in ("known", "unseen"):
        for c in bank.get(group, []):
            if c["name"] == name:
                return c.get("display", name.lower())
    return name.lower()


def prompt_ensemble(name, bank, templates=PROMPT_TEMPLATES):
    """Positive prompts for a concept (expand templates over its display phrase)."""
    disp = _display(bank, name)
    return [t.format(c=disp) for t in templates]


def negative_prompts(name, bank, templates=NEGATIVE_TEMPLATES):
    disp = _display(bank, name)
    return [t.format(c=disp) for t in templates]


def all_prompts(bank, group="known"):
    """{concept_name: {"pos": [...], "neg": [...]}} for every concept in the group."""
    out = {}
    for name in concept_names(bank, group):
        out[name] = {"pos": prompt_ensemble(name, bank), "neg": negative_prompts(name, bank)}
    return out
