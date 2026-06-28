"""The transaction input schema.

The public dataset's feature space is ``Time``, the PCA components ``V1``..``V28``
(already standardised by the dataset authors), and ``Amount``. Declaring it
explicitly gives the API validated inputs and a rich, self-documenting OpenAPI
schema at ``/docs`` instead of an opaque ``dict``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Canonical example transaction reused by OpenAPI docs and as a dashboard
# "load sample" payload. Matches the dataset's first legitimate transaction.
EXAMPLE_TRANSACTION: dict[str, float] = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62,
}


class TransactionFeatures(BaseModel):
    """A single credit-card transaction in the dataset's feature space."""

    Time: float = Field(..., description="Seconds elapsed since the first transaction.")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0.0, description="Transaction amount.")

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_TRANSACTION})
