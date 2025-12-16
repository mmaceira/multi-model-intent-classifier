# Configuration (legacy overview)

This page is kept for backwards compatibility. The configuration system has been
reorganized into layered base, provider, dataset, and experiment configs, with a
`resolved` section and derived output paths.

For the **current configuration system**, including layering rules, `resolved.*`
fields, and environment‑based selection via `DATASET`/`VARIANT`, see:

- `config.md` – Layered config, `resolved` rules, and env selection
- `output_schema.md` – How `run_id` maps to `output/runs/<label_type>/<dataset>/<variant>/`

The high‑level ideas from this page still apply (YAML + env drive datasets, models,
and experiments), but the new docs are the source of truth.
