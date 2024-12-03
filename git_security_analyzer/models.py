from pydantic import BaseModel, Field, validator
from typing import List, Optional
import json

class SecurityIssue(BaseModel):
    severity: str  # 'high', 'medium', 'low'
    description: str
    file_path: str
    line_number: Optional[int] = Field(description="Line number in the file where the vulnerability is found. Default 0.")
    code_snippet: Optional[str] = Field(description="Code snippet where the vulnerability is found.")
    recommendation: Optional[str] = Field(description="Recommendation for fixing the vulnerability.")
    class Config:
        json_encoders = {
            bool: lambda v: "True" if v else "False"
        }

class CommitAnalysis(BaseModel):
    commit_hash: str = Field(description="Commit hash")
    commit_message: str = Field(description="Commit message")
    is_security_related: bool = Field(description="Whether the commit is related to security. The option is True or False", default=False)
    is_documentation: bool = Field(description="Whether the commit is related to documentation. The option is True or False", default=False)
    is_dependency_update: bool = Field(description="Whether the commit is related to dependency update. The option is True or False", default=False)
    security_issues: List[SecurityIssue] = Field(description="Security issues found in the commit")
    analysis_summary: str = Field(description="Summary of the analysis")

    class Config:
        json_encoders = {
            bool: lambda v: "True" if v else "False"
        }

    @validator('is_security_related', 'is_documentation', 'is_dependency_update', pre=True)
    def validate_boolean(cls, v):
        if isinstance(v, str):
            return v.lower() == 'true'
        return bool(v)

