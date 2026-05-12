from pydantic import BaseModel, Field, field_validator, model_validator


class SubtitleSegment(BaseModel):
    index: int = Field(description="Subtitle segment index starting from 1.")
    start_ms: int = Field(description="Segment start time in milliseconds.")
    end_ms: int = Field(description="Segment end time in milliseconds.")
    text: str = Field(description="Subtitle text.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Subtitle text must not be empty.")
        return value

    @model_validator(mode="after")
    def validate_time_range(self) -> "SubtitleSegment":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms.")
        return self


class SubtitleResult(BaseModel):
    language: str = Field(description="Detected main language code, for example ko-KR.")
    summary: str = Field(description="Short summary of the spoken answer.")
    segments: list[SubtitleSegment] = Field(description="Generated subtitle segments.")