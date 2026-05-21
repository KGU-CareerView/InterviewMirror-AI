from pydantic import BaseModel, Field


class ReportStrength(BaseModel):
    title: str = Field(description="Strength title.")
    detail: str = Field(description="Detailed explanation.")


class ReportWeakness(BaseModel):
    title: str = Field(description="Weakness title.")
    detail: str = Field(description="Detailed explanation.")
    improvement: str = Field(description="Concrete improvement suggestion.")


class ReportTimeInsight(BaseModel):
    time_range: str = Field(description="Time range, for example 00:30-01:00.")
    observation: str = Field(description="Observed behavior or score pattern.")
    suggestion: str = Field(description="Suggestion for this time range.")


class FinalInterviewReport(BaseModel):
    overall_summary: str = Field(description="Overall interview summary.")
    overall_score: float = Field(description="Final score from 0 to 100.")

    content_score: float = Field(description="Answer content score from 0 to 100.")
    voice_score: float = Field(description="Voice tone score from 0 to 100.")
    expression_score: float = Field(description="Facial expression score from 0 to 100.")

    strengths: list[ReportStrength] = Field(description="Main strengths.")
    weaknesses: list[ReportWeakness] = Field(description="Main weaknesses.")
    time_based_insights: list[ReportTimeInsight] = Field(description="Time-based feedback.")

    final_advice: str = Field(description="Final interview improvement advice.")
