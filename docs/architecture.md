# Architecture

This document describes the system architecture, the ML pipeline, the layered
serving backend, and the analytics dashboard of the Credit Card Fraud Detection
project.

The system has **three decoupled layers**:

1. **Training pipeline** (`src/`, `main.py`) — produces the production artifacts.
2. **REST API** (`app/`) — a layered FastAPI service that *loads* and serves the
   model (never retrains).
3. **Dashboard** (`dashboard/`) — a Dash + Plotly frontend that consumes the API
   over HTTP.

## System overview

```mermaid
flowchart TB
    subgraph Sources["Data sources"]
        K[("Kaggle\nmlg-ulb/creditcardfraud")]
        S["Synthetic generator\n(offline fallback)"]
    end

    subgraph Config["Configuration"]
        Y["configs/config.yaml"]
    end

    subgraph Pipeline["Training pipeline (main.py)"]
        L["Loader\n(real or synthetic)"]
        SP["Stratified split"]
        T["Optuna tuning\n(optional)"]
        TR["Train 5 models\n(imblearn Pipeline)"]
        EV["Evaluate\n(imbalance-aware)"]
        SEL["Select best\n(by PR-AUC)"]
        EX["SHAP explainability"]
    end

    subgraph Artifacts["Artifacts"]
        M["models/best_model.joblib\n+ metadata"]
        R["reports/*\n(plots, CSV, JSON)"]
        ML["MLflow runs\n(mlruns/)"]
    end

    subgraph Serving["Serving"]
        API["FastAPI service\nrouters/services/schemas/core"]
        DASH["Dash + Plotly\ndashboard"]
    end

    K --> L
    S --> L
    Y --> Pipeline
    L --> SP --> T --> TR --> EV --> SEL --> EX
    SEL --> M
    EV --> R
    EX --> R
    TR --> ML
    M --> API
    R -. static images/metrics .-> API
    API -->|HTTP| DASH
```

## ML pipeline (leakage-free)

Every model is wrapped in an `imblearn` `Pipeline` so that scaling and SMOTE are
fit **inside** cross-validation folds and on the training split only. SMOTE acts
during `fit` and is automatically a no-op at inference, so the persisted pipeline
is directly deployable.

```mermaid
flowchart LR
    X["Raw features\nTime, V1..V28, Amount"] --> P["ColumnTransformer\nStandardScaler(Time, Amount)\npassthrough V1..V28"]
    P --> SM["SMOTE\n(fit-time only)"]
    SM --> C["Classifier\nLR / RF / XGB / LGBM / CatBoost"]
    C --> O["Fraud probability"]
    O --> TH["Threshold\n(max-F1 on PR curve)"]
    TH --> D["is_fraud"]
```

## Why these choices

| Decision | Rationale |
| --- | --- |
| **PR-AUC for selection** | On a ~0.17% positive class, accuracy and even ROC-AUC are misleading; PR-AUC reflects performance on the rare positive class. |
| **SMOTE inside the pipeline** | Prevents test-set leakage and makes the saved artifact self-contained. |
| **Threshold tuning** | SMOTE rebalances the training distribution, miscalibrating probabilities; the default 0.5 cutoff is a poor operating point. |
| **Synthetic fallback** | Keeps the project runnable offline (CI, fresh clones) with zero credentials, without removing real-data support. |
| **Config-driven** | A single `config.yaml` makes runs reproducible and removes magic numbers from code. |

## Serving architecture (layered backend)

The API follows a layered architecture so each concern has a single home and
routers stay thin (parse → call a service → return a schema):

```mermaid
flowchart TB
    R["routers/\nmeta · health · prediction · model"]
    D["dependencies.py\n(FastAPI DI providers)"]
    S["services/\nPredictionService · ModelService · PredictionHistory"]
    SC["schemas/\nPydantic request/response models"]
    C["core/\nsettings · middleware · exception handlers · logging"]
    ST["startup.py\nAppState singleton (model + artifacts)"]
    FP["src/predict.py\nFraudPredictor"]

    R --> D --> S --> FP
    R --> SC
    C -. wraps every request .-> R
    ST -. provides state .-> D
```

| Layer | Responsibility |
| --- | --- |
| `core/` | Settings (`pydantic-settings`), request-context middleware (request id + latency), centralized exception handlers, logging. |
| `schemas/` | Validated, self-documenting request/response models (rich OpenAPI). |
| `services/` | Business logic: enrich predictions, read report artifacts, track history. |
| `routers/` | HTTP endpoints grouped by concern; no business logic. |
| `dependencies.py` | Dependency-injection seam wiring services from `AppState`. |
| `startup.py` | Loads the model + report artifacts **once** at startup into a process-wide singleton. |

## Request flow (serving)

```mermaid
sequenceDiagram
    participant Client as Client / Dashboard
    participant MW as Middleware (req-id, timing)
    participant API as Router
    participant DI as Dependencies
    participant PS as PredictionService
    participant Model as best_model.joblib

    Note over API,Model: On startup: load model + artifacts once into AppState
    Client->>MW: POST /predict {transaction}
    MW->>API: attach request-id + start timer
    API->>DI: require_model()
    alt model loaded
        DI->>PS: PredictionService(predictor, history)
        PS->>Model: predict_proba(features)
        Model-->>PS: probability
        PS->>PS: enrich (label, confidence, risk, latency, timestamp) + record history
        PS-->>API: enriched prediction
        API-->>MW: 200 JSON
        MW-->>Client: 200 + X-Request-ID + X-Process-Time-Ms
    else model missing
        DI-->>Client: 503 ModelNotFoundError (uniform error envelope)
    end
```

## Dashboard

The dashboard is a separate process that never imports the model or the `app`
package — all data flows through the REST API, exactly as a real frontend would.

```mermaid
flowchart LR
    subgraph Dashboard["Dash + Plotly (:8050)"]
        AC["api_client.py\n(requests wrapper)"]
        SEC["sections.py\nOverview · Predict · Model · History"]
        CH["components/charts.py\n(pure Plotly figures)"]
        CB["callbacks/\nnavigation · prediction · theme"]
    end
    AC -->|HTTP /metrics /model-info\n/feature-importance /history /predict| API["FastAPI :8000"]
    API -->|/static/reports/*.png| SEC
    CB --> SEC --> CH
```

Design notes:

- **Pure figure builders** (`components/charts.py`) take data + a theme and return
  Plotly figures with no network access — making them unit-testable.
- **Hybrid charts** — interactive Plotly for metrics, feature importance, class
  distribution, and the live history stream; the ROC / PR / confusion-matrix PNGs
  produced by the training pipeline are served by the API as static files and
  embedded directly.
- **Themes** — a dark/light toggle writes to a `dcc.Store`; CSS variables in
  `assets/styles.css` re-cascade and charts re-render with matching colours.
- **Resilience** — every API call returns `(data, error)`; a backend hiccup shows
  a banner instead of crashing the UI.
