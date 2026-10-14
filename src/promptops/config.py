from __future__ import annotations

import boto3
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROMPTOPS_", extra="ignore")

    prompt_name: str = "document-intake-triage"
    default_ref: str = "prod"
    default_model: str = "nova-lite"
    s3_bucket: str = ""
    aws_profile: str | None = Field(default=None, validation_alias="AWS_PROFILE")
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")

    def session(self) -> boto3.Session:
        return boto3.Session(profile_name=self.aws_profile, region_name=self.aws_region)


def get_settings() -> Settings:
    return Settings()
