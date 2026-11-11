from pydantic import BaseModel, Field


class VariableOut(BaseModel):
    code: str
    name: str
    definition: str = Field(description="Plain-language definition for students and the public")
    default_unit: str = Field(description="Catalog unit; each series reports the unit NASA actually returned")
    category: str
    is_reanalysis: bool = Field(description="True if model-assimilated rather than a direct measurement")
    dataset_note: str
    coverage_start_year: int
    percent_change_meaningful: bool = Field(
        description="False for scales without a true zero (e.g. °C), where percent change is not reported"
    )
