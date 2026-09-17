"""CI drift gate for the M2 model registry.

Proves, on every push, that the governance rules still hold:

1. the default registry is valid and every seeded model is approved;
2. the committed `models/manifest.json` has not drifted from the code;
3. the live-path gate REFUSES unregistered / candidate / retired models
   (the guard is not vacuous);
4. an override works but is recorded;
5. promotion refuses without OOS evidence, with a losing candidate, or with
   an automated approver;
6. retirement never deletes an entry or its artifact;
7. the orchestrator holds no hardcoded model-version literals.

Exit 1 on any drift, with a diff-precise message.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    MODEL_STATUS_APPROVED,
    MODEL_STATUS_CANDIDATE,
    MODEL_STATUS_RETIRED,
)
from core.model_registry import (  # noqa: E402
    ModelEntry,
    ModelNotApprovedError,
    ModelRegistryError,
    build_default_model_registry,
    load_model_manifest,
    require_live_model,
)

_APPROVED_AT = "2026-09-16T00:00:00+00:00"


def _winning(incumbent: str) -> dict:
    return {
        "primary_metric": "oos_sharpe",
        "candidate_value": 1.4,
        "incumbent_value": 1.1,
        "sample": 500,
        "candidate_version": "gate-candidate-v1",
        "incumbent_version": incumbent,
    }


def main() -> int:
    failures: list[str] = []
    registry = build_default_model_registry()

    # 1. Registry validity.
    problems = registry.problems()
    if problems:
        failures.append(f"default registry is invalid: {problems[:3]}")
    for version, entry in registry.all_models().items():
        if entry.status != MODEL_STATUS_APPROVED:
            failures.append(f"seeded model {version} is {entry.status}, expected approved")

    # 2. Committed manifest must match the code.
    manifest = load_model_manifest()
    if not manifest:
        failures.append("models/manifest.json is missing — the registry must be committed")
    elif manifest.get("registry_hash") != registry.registry_hash():
        failures.append(
            "models/manifest.json has drifted from the code:\n"
            f"    manifest {manifest.get('registry_hash')}\n"
            f"    code     {registry.registry_hash()}\n"
            "    re-run persist_model_registry(build_default_model_registry())"
        )

    # 3. The live gate must refuse. A gate that cannot fail is theatre.
    def _refuses(version: str, label: str) -> None:
        try:
            require_live_model(version, registry)
        except ModelNotApprovedError:
            return
        failures.append(f"live gate did NOT refuse a {label} model ({version})")

    probe = build_default_model_registry()
    probe.register(ModelEntry(
        model_version="gate-candidate-v1",
        family="technical_analysis",
        feature_set_version="fs-1",
    ))
    probe.retire("fundamental-v1", _APPROVED_AT, "gate probe")

    _refuses("definitely-not-registered-v9", "unregistered")
    try:
        require_live_model("gate-candidate-v1", probe)
        failures.append("live gate did NOT refuse a candidate model")
    except ModelNotApprovedError:
        pass
    try:
        require_live_model("fundamental-v1", probe)
        failures.append("live gate did NOT refuse a retired model")
    except ModelNotApprovedError:
        pass

    # 4. Override works, and is recorded.
    record = require_live_model(
        "definitely-not-registered-v9", registry, override_reason="ci gate probe"
    )
    if not record.get("override") or record.get("override_reason") != "ci gate probe":
        failures.append("an override did not record itself in the resolution record")

    # 5. Promotion gates.
    incumbent = probe.incumbent("technical_analysis")
    incumbent_version = incumbent.model_version if incumbent else None
    for label, kwargs in (
        ("no OOS comparison", dict(oos_comparison={}, approved_by="oren")),
        (
            "losing candidate",
            dict(
                oos_comparison={**_winning(incumbent_version), "candidate_value": 0.5},
                approved_by="oren",
            ),
        ),
        (
            "automated approver",
            dict(oos_comparison=_winning(incumbent_version), approved_by="ci"),
        ),
    ):
        try:
            probe.promote(
                "gate-candidate-v1",
                kwargs["approved_by"],
                _APPROVED_AT,
                kwargs["oos_comparison"],
            )
            failures.append(f"promotion was allowed with {label}")
        except ModelRegistryError:
            pass

    # 6. Retirement never deletes.
    retired = probe.get("fundamental-v1")
    if retired is None:
        failures.append("retirement DELETED the entry — artifacts must survive")
    elif retired.status != MODEL_STATUS_RETIRED:
        failures.append(f"retired model has status {retired.status}")

    # 6b. M5 artifact provenance: a trained artifact must be reproducible.
    from types import SimpleNamespace

    from core.config import CALIBRATION_UNCALIBRATED
    from core.model_registry import entry_from_training_run, entry_problems

    def _trained(**overrides) -> ModelEntry:
        payload = dict(
            model_version="gate-trained-v1", family="boosting",
            feature_set_version="fs-gate", dataset_hash="d" * 64,
            code_commit="abc1234", hyperparameters={"n_estimators": 100},
            seed=42, artifact_hash="a" * 64, horizon="20d",
            universe="universe-ledger-v1",
        )
        payload.update(overrides)
        return ModelEntry(**payload)

    if entry_problems(_trained()):
        failures.append(
            f"a fully-provenanced trained entry was rejected: {entry_problems(_trained())[:2]}"
        )
    for missing in ("code_commit", "dataset_hash", "horizon", "universe"):
        if not entry_problems(_trained(**{missing: None})):
            failures.append(
                f"a trained artifact missing {missing} was accepted — it could "
                f"not be reproduced or audited"
            )
    if not entry_problems(_trained(seed=None)):
        failures.append("a trained artifact without a seed was accepted")
    if not entry_problems(_trained(hyperparameters={})):
        failures.append("a trained artifact without hyperparameters was accepted")
    if not entry_problems(_trained(forecast_target="vibes")):
        failures.append("an unknown forecast_target was accepted")
    if not entry_problems(_trained(calibration_version="")):
        failures.append(
            "a blank calibration_version was accepted — 'uncalibrated' is "
            "honest, a missing field is not"
        )

    # Deterministic scorers must stay exempt: they have no artifact.
    for entry in registry.all_models().values():
        if entry.is_trained():
            failures.append(f"seeded scorer {entry.model_version} claims to be trained")
        if entry_problems(entry):
            failures.append(f"seeded scorer {entry.model_version}: {entry_problems(entry)[:1]}")

    # Provenance must be CAPTURED from a run, not retyped.
    synthetic_run = SimpleNamespace(
        model_family="boosting", dataset_hash="d" * 64, feature_set_hash="f" * 64,
        artifact_hash="a" * 64, seed=42, hyperparameters={"n_estimators": 100},
        target_horizon="20d", metrics={"rmse": 0.1},
        folds=[SimpleNamespace(validation_start_time="2026-01-05 00:00:00")],
    )
    captured = entry_from_training_run(synthetic_run, "gate-captured-v1", universe="u-v1")
    if captured.status != MODEL_STATUS_CANDIDATE:
        failures.append(
            f"a model built from a training run is born {captured.status!r}; it "
            f"must be a candidate — registering is not trusting"
        )
    for label, expected, actual in (
        ("dataset_hash", synthetic_run.dataset_hash, captured.dataset_hash),
        ("artifact_hash", synthetic_run.artifact_hash, captured.artifact_hash),
        ("seed", synthetic_run.seed, captured.seed),
        ("horizon", synthetic_run.target_horizon, captured.horizon),
    ):
        if expected != actual:
            failures.append(f"entry_from_training_run did not capture {label}")
    if not captured.code_commit:
        failures.append("entry_from_training_run did not capture code_commit")
    if captured.calibration_version != CALIBRATION_UNCALIBRATED:
        failures.append("a fresh artifact is not marked uncalibrated")

    # 7. No hardcoded literals in the orchestrator.
    source = (REPO_ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    for literal in ('model_version="market-data-v1"',
                    'model_version="technical-v1"',
                    'model_version="fundamental-v1"'):
        if literal in source:
            failures.append(f"orchestrator still hardcodes {literal} — must be a registry lookup")

    if failures:
        print("M2 model-registry gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("M2 model-registry gate OK:")
    print(
        f"  {len(registry.all_models())} models registered, all approved; "
        f"manifest matches code ({registry.registry_hash()[:16]}...)"
    )
    print("  live gate refuses unregistered/candidate/retired; override is audited.")
    print("  promotion refuses without OOS evidence, on a loss, or without a human approver.")
    print("  retirement preserves entries; orchestrator holds no hardcoded versions.")
    print("  M5: trained artifacts must carry full provenance; run provenance is captured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
