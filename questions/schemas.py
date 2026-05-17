from pydantic import BaseModel, Field, field_validator, model_validator


class QuestionItemModel(BaseModel):
    index: int = Field(description="Question index starting from 1.")
    question: str = Field(description="Generated interview question.")
    tooltip: str = Field(description="Short helpful tooltip for answering the question.")
    category: str = Field(description="Question category.")
    intent: str = Field(description="Why this question is being asked.")
    answer_keywords: list[str] = Field(description="Useful keywords or points the interviewee should include.")

    @field_validator("question", "tooltip", "category", "intent")
    @classmethod
    def validate_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text field must not be empty.")
        return value

    @field_validator("answer_keywords")
    @classmethod
    def validate_keywords(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("answer_keywords must not be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_index(self) -> "QuestionItemModel":
        if self.index < 1:
            raise ValueError("index must start from 1.")
        return self


class QuestionGenerateResult(BaseModel):
    questions: list[QuestionItemModel] = Field(description="Generated interview questions with tooltips.")

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, value: list[QuestionItemModel]) -> list[QuestionItemModel]:
        if not value:
            raise ValueError("questions must not be empty.")
        return value
