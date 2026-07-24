from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import requests

from config import get_supabase_key, get_supabase_url


def is_supabase_available() -> bool:
    return bool(get_supabase_url()) and bool(get_supabase_key())


def get_supabase_status_message() -> str:
    if not get_supabase_url():
        return "Falta configurar a URL do Supabase. Aceita `SUPABASE_URL` ou `[supabase].url`."
    if not get_supabase_key():
        return (
            "Falta configurar a chave do Supabase. Aceita `SUPABASE_KEY`, `SUPABASE_ANON_KEY`, "
            "`SUPABASE_PUBLISHABLE_KEY` ou `[supabase].key`."
        )
    return "Supabase configurado."


def _build_user(data: dict[str, Any] | None) -> SimpleNamespace | None:
    if not data:
        return None
    return SimpleNamespace(
        id=data.get("id"),
        email=data.get("email"),
        user_metadata=data.get("user_metadata") or {},
    )


def _build_session(data: dict[str, Any] | None) -> SimpleNamespace | None:
    if not data:
        return None
    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    if not access_token or not refresh_token:
        return None
    return SimpleNamespace(access_token=access_token, refresh_token=refresh_token)


def _auth_headers(access_token: str | None = None) -> dict[str, str]:
    api_key = get_supabase_key()
    headers = {
        "apikey": api_key or "",
        "Content-Type": "application/json",
    }
    headers["Authorization"] = f"Bearer {access_token or api_key or ''}"
    return headers


def _rest_headers(access_token: str | None = None, prefer: str | None = None) -> dict[str, str]:
    headers = _auth_headers(access_token)
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _raise_for_supabase_error(response: requests.Response) -> None:
    try:
        payload = response.json()
    except Exception:
        payload = response.text

    if response.ok:
        return

    if isinstance(payload, dict):
        message = payload.get("msg") or payload.get("message") or payload.get("error_description") or payload.get("error")
        if message:
            raise RuntimeError(str(message))
    raise RuntimeError(str(payload))


def sign_in_with_password(email: str, password: str) -> tuple[SimpleNamespace | None, SimpleNamespace | None]:
    url = get_supabase_url()
    response = requests.post(
        f"{url}/auth/v1/token?grant_type=password",
        headers=_auth_headers(),
        json={"email": email, "password": password},
        timeout=15,
    )
    _raise_for_supabase_error(response)
    data = response.json()
    return _build_user(data.get("user")), _build_session(data)


def sign_up(email: str, password: str, user_name: str) -> tuple[SimpleNamespace | None, SimpleNamespace | None]:
    url = get_supabase_url()
    response = requests.post(
        f"{url}/auth/v1/signup",
        headers=_auth_headers(),
        json={
            "email": email,
            "password": password,
            "data": {"user_name": user_name},
        },
        timeout=15,
    )
    _raise_for_supabase_error(response)
    data = response.json()
    return _build_user(data.get("user")), _build_session(data.get("session"))


def sign_out(access_token: str | None) -> None:
    if not access_token or not get_supabase_url():
        return
    try:
        requests.post(
            f"{get_supabase_url()}/auth/v1/logout",
            headers=_auth_headers(access_token),
            timeout=15,
        )
    except Exception:
        pass


def upsert_profile(profile_payload: dict[str, Any], access_token: str) -> None:
    response = requests.post(
        f"{get_supabase_url()}/rest/v1/profiles?on_conflict=id",
        headers=_rest_headers(access_token, prefer="resolution=merge-duplicates,return=representation"),
        json=profile_payload,
        timeout=15,
    )
    _raise_for_supabase_error(response)


def fetch_profile_user_name(user_id: str, access_token: str) -> str | None:
    response = requests.get(
        f"{get_supabase_url()}/rest/v1/profiles",
        headers=_rest_headers(access_token),
        params={"select": "user_name", "id": f"eq.{quote(user_id)}", "limit": 1},
        timeout=15,
    )
    _raise_for_supabase_error(response)
    rows = response.json() or []
    if not rows:
        return None
    user_name = (rows[0].get("user_name") or "").strip()
    return user_name or None


def insert_simulation(payload: dict[str, Any], access_token: str) -> None:
    response = requests.post(
        f"{get_supabase_url()}/rest/v1/simulations",
        headers=_rest_headers(access_token, prefer="return=representation"),
        json=payload,
        timeout=15,
    )
    _raise_for_supabase_error(response)


def fetch_simulations(limit: int, client_id: str, access_token: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{get_supabase_url()}/rest/v1/simulations",
        headers=_rest_headers(access_token),
        params={
            "select": "*",
            "client_id": f"eq.{quote(client_id)}",
            "order": "created_at.desc",
            "limit": limit,
        },
        timeout=15,
    )
    _raise_for_supabase_error(response)
    return response.json() or []