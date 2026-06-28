# Architecture

This document describes the system architecture, the ML pipeline, and the
serving path of the Credit Card Fraud Detection project.

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
        API["FastAPI service\n/predict /health"]
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

## Request flow (serving)

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant State as AppState (singleton)
    participant Model as best_model.joblib

    Note over API,Model: On startup: load model + metadata once
    Client->>API: POST /predict {transaction}
    API->>State: predictor available?
    alt model loaded
        State->>Model: predict_proba(features)
        Model-->>State: probability
        State-->>API: {is_fraud, fraud_probability, threshold, model_name}
        API-->>Client: 200 JSON
    else model missing
        API-->>Client: 503 Service Unavailable
    end
```
