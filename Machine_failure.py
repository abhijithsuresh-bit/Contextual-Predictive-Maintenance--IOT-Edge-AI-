from pydantic import BaseModel, Field


class Machine_failure(BaseModel):
    Product_ID: str
    Type: str

    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: int
    Torque_Nm: float
    Tool_wear_min: int

    # ------------------------------------------------------------------
    # LEAKAGE FLAGS - delete this whole block once you retrain.
    #
    # In the AI4I 2020 dataset these five columns ARE the failure modes,
    # and the "Machine failure" label is essentially their logical OR.
    # They are not known at prediction time, so the UI always sends 0.
    # Defaults are set here so the API still works if they are omitted.
    # ------------------------------------------------------------------
    TWF: int = Field(default=0)
    HDF: int = Field(default=0)
    PWF: int = Field(default=0)
    OSF: int = Field(default=0)
    RNF: int = Field(default=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Product_ID": "M14860",
                    "Type": "M",
                    "Air_temperature_K": 298.1,
                    "Process_temperature_K": 308.6,
                    "Rotational_speed_rpm": 1551,
                    "Torque_Nm": 42.8,
                    "Tool_wear_min": 108,
                }
            ]
        }
    }