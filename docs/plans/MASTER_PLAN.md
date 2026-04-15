# Diamond Analytics — Proprietary Models Master Plan

## Overview
16 proprietary analytics models across 4 domains, built to production standards.
**No mock data. Full test coverage. Data validation on every boundary.**

---

## Implementation Phases

### Phase 0: Infrastructure (MUST BUILD FIRST)
Everything depends on this. Testing framework, base classes, validation layer.

| Component | Files | Purpose |
|-----------|-------|---------|
| pytest config | `pyproject.toml`, `conftest.py`, `tests/conftest.py` | Test framework setup |
| Test fixtures | `tests/fixtures/sample_data.py` | Real data sampling from DB (not mocks) |
| Base classes | `src/analytics/base.py` | `BaseAnalyticsModel` with train/predict/evaluate/validate |
| Validation | `src/analytics/validation.py` | Pydantic schemas for all inputs/outputs |
| Feature eng | `src/analytics/features.py` | Shared feature extractors (z-scores, rolling, decay) |
| Model registry | `src/analytics/registry.py` | Versioned model storage with metadata |
| Query layer | `src/db/model_queries.py` | Parameterized batch queries for model training |

### Phase 1: Quick Wins (sklearn/scipy only — no new deps)
These use tools already installed. Highest impact, fastest to ship.

| # | Model | Domain | New Deps | Est. Complexity |
|---|-------|--------|----------|----------------|
| 1 | **CausalWAR** | ML/AI | econml | Low-Medium |
| 2 | **Player Sharpe Ratio** | Finance | (none) | Low |
| 3 | **Kinetic Half-Life (K½)** | Biomech | (none) | Medium |
| 4 | **MESI** | Biomech | (none) | Medium |
| 5 | **PIVS** | Finance | (none) | Medium |
| 6 | **PSET** | Cross-sport | (none) | Medium |

### Phase 2: Advanced Models (new deps, more complexity)

| # | Model | Domain | New Deps | Est. Complexity |
|---|-------|--------|----------|----------------|
| 7 | **Pitch Sequence Alpha Decay** | Finance | hmmlearn | Medium-High |
| 8 | **LOFT** | Finance | (none) | Medium |
| 9 | **PDR** | Cross-sport | ruptures | Medium-High |
| 10 | **ABL** | Biomech | (none) | Medium |
| 11 | **VWR** | Biomech | (none) | Medium |
| 12 | **BGI** | Cross-sport | econml | High |

### Phase 3: Deep Learning Models (PyTorch required)

| # | Model | Domain | New Deps | Est. Complexity |
|---|-------|--------|----------|----------------|
| 13 | **MechanixAE** | ML/AI | pytorch, ruptures | High |
| 14 | **PitchGPT** | ML/AI | pytorch | Very High |
| 15 | **ChemNet** | ML/AI | pytorch-geometric | Very High |
| 16 | **DPI** | Cross-sport | (none, but needs tracking data) | High |

---

## Quality Standards

### Testing Requirements
- **Unit tests**: Every public function in every model module
- **Integration tests**: Full train → predict → validate pipeline per model
- **Data tests**: Schema validation, null rate checks, range validation
- **Regression tests**: Model output stability across code changes
- **Edge cases**: Empty data, single pitch, missing columns, single-season pitcher
- **No mock data**: All tests use sampled real Statcast data via fixtures
- **Target**: 90%+ coverage on analytics modules

### Data Validation
- Pydantic schemas for every model input and output
- Decorator-based validation on all public functions
- Column presence + dtype + range checks on all Statcast queries
- Null rate thresholds (reject data with >20% nulls on critical fields)
- Distribution drift detection on model inputs vs training distribution

### Model Registry
- Every trained model gets: version hash, training date, data date range, hyperparams, eval metrics
- Models stored as `models/{model_name}_v{version}.pkl` with `models/{model_name}_meta.json`
- Lazy loading with LRU cache for serving

### Code Standards
- Type hints on all function signatures
- Docstrings on all public functions
- `from __future__ import annotations` in every file
- Logging via `logging.getLogger(__name__)`
- Graceful import pattern in dashboard views

---

## Dependency Additions

### Phase 1
```
econml>=0.15
xgboost>=2.0
```

### Phase 2
```
hmmlearn>=0.3
ruptures>=1.1
```

### Phase 3
```
torch>=2.0
torch-geometric>=2.4  (Phase 3 only)
```

### Testing
```
pytest>=8.0
pytest-cov>=5.0
pytest-xdist>=3.0  (parallel test execution)
```

---

## Dashboard Integration

Each model gets its own Streamlit view in `src/dashboard/views/` and a nav entry in `app.py`.

New navigation structure:
```
Navigation:
├── Live Game
├── Matchup Explorer
├── Bullpen Strategy
├── Anomaly Alerts
├── Stuff+
├── Sequencing
├── Phillies Hub
├── Data Management
├── ── Proprietary Models ──
├── CausalWAR
├── Volatility Surface (PIVS)
├── Lineup Optimizer (Sharpe)
├── Alpha Decay
├── LOFT (Order Flow)
├── Pitch Decay (K½)
├── Batting Load (ABL)
├── Arm Stress (VWR)
├── Pitch Stability (MESI)
├── Pitch Threat (PSET)
├── Runner Gravity (BGI)
├── PitchGPT
├── Lineup Chemistry (ChemNet)
├── MechanixAE
└── Defensive Pressing (DPI)
```

---

## Execution Strategy

1. **Infrastructure agent** builds Phase 0 (testing + base classes + validation)
2. **Model agents** build Phase 1 models in parallel (6 agents, one per model)
3. Each model agent: writes analytics module → writes tests → runs tests → writes dashboard view
4. Integration agent wires all new views into app.py navigation
5. Repeat for Phase 2 and Phase 3
