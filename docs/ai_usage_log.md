# AI usage log

Disclosure log of AI-assisted work on this TFM, per UNED's academic
integrity guidance on tool use. One entry per work session; append, don't
rewrite history.

## 2026-08-10 — Data pipeline (Stage 1: data gathering)

- **Tool**: Claude Code (Anthropic).
- **Scope**: Wrote `src/data_pipeline.py` (download S&P 500 `^GSPC` OHLCV
  via `yfinance`, clean/flag, compute log returns), `requirements.txt`,
  `README.md`, and the auto-generated `docs/data_provenance.md`.
- **Environment troubleshooting done by the assistant**: the dev machine's
  `yfinance` install (0.1.74, on Python 3.7.4) could not reach Yahoo's
  current API. Diagnosed and resolved by pinning `yfinance==0.2.55` and
  `multitasking==0.0.11` (both compatible with Python 3.7) — see comments
  in `requirements.txt`.
- **Human review**: outputs (raw/processed CSVs, provenance note, flagged
  rows — a 2020-03-16 COVID-crash outlier return and a 2023-05-24
  zero-volume day) were inspected and spot-checked against known market
  events before accepting.
- **Not AI-generated**: research questions, methodology decisions (walk-
  forward scheme, model list, VaR/ES approach, feature-set separation),
  and all judgment calls in `CLAUDE.md` — these are the author's, set
  before this session and given to the assistant as a brief.
