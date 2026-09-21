from enum import Enum

from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFORMATIONAL = "informational"


class DeviationType(str, Enum):
    ELIGIBILITY = "eligibility_violation"
    DOSING = "dosing_deviation"
    VISIT_SCHEDULE = "visit_schedule_deviation"
    CONSENT = "informed_consent_issue"
    SAFETY_REPORTING = "safety_reporting_gap"
    DATA_COLLECTION = "data_collection_deviation"
    PROCEDURE = "procedure_deviation"
    OTHER = "other"


class DeviationFinding(BaseModel):
    deviation_type: DeviationType
    severity: SeverityLevel
    description: str = Field(..., description="What the deviation is")
    protocol_reference: str = Field(
        ..., description="Which protocol section/rule is violated"
    )
    evidence: str = Field(
        ...,
        description="Evidence from the retrieved context supporting this finding",
    )
    recommendation: str = Field(
        ..., description="Corrective or preventive action"
    )


class DeviationAnalysisOutput(BaseModel):
    """Schema for structured LLM output via ChatOpenAI.with_structured_output()."""

    status: str = Field(
        ..., description="'complete', 'needs_more_context', or 'failed'"
    )
    reasoning: str = Field(..., description="Chain-of-thought analysis")
    missing_info: str = Field(
        default="", description="What data is still needed, if any"
    )
    findings: list[DeviationFinding] = Field(default_factory=list)


class IntentClassification(BaseModel):
    """Schema for the intent classifier's structured output."""

    mode: str = Field(
        ...,
        description="'deviation_analysis' if the user describes a clinical scenario to check, "
        "'conversational' if this is a follow-up question, clarification, or general query",
    )
    rewritten_query: str = Field(
        ...,
        description="Self-contained version of the user's query with all anaphora and "
        "implicit references resolved using conversation context. If the query is "
        "already self-contained, return it unchanged.",
    )
