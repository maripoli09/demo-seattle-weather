import os
from typing import Any

import streamlit as st


def _mapping_get(mapping: Any, key: str) -> str | None:
    if mapping is None:
        return None
    if hasattr(mapping, "get"):
        value = mapping.get(key)
    else:
        try:
            value = mapping[key]
        except Exception:
            value = None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def get_secret_or_env(
    names: list[str],
    *,
    section_names: list[str] | None = None,
    field_names: list[str] | None = None,
) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()

    for name in names:
        value = _mapping_get(st.secrets, name)
        if value:
            return value

    for section_name in section_names or []:
        section = _mapping_get(st.secrets, section_name)
        if isinstance(section, str):
            return section
        if section is None:
            continue
        for field_name in field_names or []:
            value = _mapping_get(section, field_name)
            if value:
                return value

    return None


def get_supabase_url() -> str | None:
    return get_secret_or_env(
        ["SUPABASE_URL", "SUPABASE_PROJECT_URL"],
        section_names=["supabase", "SUPABASE"],
        field_names=["url", "project_url", "SUPABASE_URL"],
    )


def get_supabase_key() -> str | None:
    return get_secret_or_env(
        [
            "SUPABASE_KEY",
            "SUPABASE_ANON_KEY",
            "SUPABASE_PUBLISHABLE_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
        ],
        section_names=["supabase", "SUPABASE"],
        field_names=[
            "key",
            "anon_key",
            "publishable_key",
            "service_role_key",
            "SUPABASE_KEY",
        ],
    )


def get_openweather_api_key() -> str | None:
    return get_secret_or_env(
        ["OPENWEATHER_API_KEY"],
        section_names=["openweather", "OPENWEATHER"],
        field_names=["api_key", "OPENWEATHER_API_KEY"],
    )