# ML Governance Module

This module owns the **authoritative registry** for machine learning datasets.

## Responsibilities
- Dataset registration
- Hash-based deduplication
- Dataset lineage
- Audit logging
- Long-term governance

## Backing Store
- PostgreSQL schema: `ml_governance`

## Design Principles
- Datasets are immutable once registered
- Content hash is the source of truth
- MLflow consumes metadata but does not own it

## Typical Flow
1. Dataset generated (JSONL)
2. Dataset registered via `DatasetRegistry`
3. Training run references dataset hash
4. Audit trail records usage

This module should never contain:
- Training logic
- Pipeline transforms
- Ephemeral or scratch data
