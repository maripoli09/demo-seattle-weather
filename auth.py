from typing import Any

import streamlit as st

try:
    from supabase import create_client
except ModuleNotFoundError:
    create_client = None


AUTH_DEFAULTS = {
    "user": None,
    "access_token": None,
    "refresh_token": None,
    "user_name": "namename",
}


def init_auth_state() -> None:
    """Initialize auth keys in Streamlit session state."""
    for key, value in AUTH_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_supabase_client(authenticated: bool = False) -> Any | None:
    """Return a Supabase client, optionally with user session attached."""
    if create_client is None:
        return None

    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        return None

    supabase = create_client(url, key)

    if authenticated:
        access_token = st.session_state.get("access_token")
        refresh_token = st.session_state.get("refresh_token")
        if not access_token or not refresh_token:
            return None
        supabase.auth.set_session(access_token, refresh_token)

    return supabase


def ensure_profile_exists() -> tuple[bool, str]:
    """Upsert the authenticated user profile in the profiles table."""
    user = st.session_state.get("user")
    if user is None:
        return False, "Invalid session."

    supabase = get_supabase_client(authenticated=True)
    if supabase is None:
        return False, "Invalid session or Supabase is not configured."

    try:
        user_name = st.session_state.get("user_name") or "namename"
        supabase.table("profiles").upsert(
            {"id": user.id, "user_name": user_name}, on_conflict="id"
        ).execute()
        return True, ""
    except Exception as exc:
        return False, f"Error ensuring profile: {exc}"


def login(email: str, password: str) -> tuple[bool, str]:
    """Authenticate a user and store tokens in session state."""
    if not email or not password:
        return False, "Email and password are required."

    supabase = get_supabase_client()
    if supabase is None:
        return False, "Supabase is not configured."

    try:
        response = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )

        if response.user is None or response.session is None:
            return False, "Authentication failed."

        st.session_state.user = response.user
        st.session_state.access_token = response.session.access_token
        st.session_state.refresh_token = response.session.refresh_token
        st.session_state.user_name = (
            getattr(response.user, "user_metadata", {}) or {}
        ).get("user_name", "namename")

        ok_profile, profile_msg = ensure_profile_exists()
        if not ok_profile:
            return False, profile_msg

        return True, f"Welcome, {response.user.email}!"
    except Exception as exc:
        return False, f"Login error: {exc}"


def register(email: str, password: str) -> tuple[bool, str]:
    """Create a user account and store tokens when a session is returned."""
    if not email or not password:
        return False, "Email and password are required."

    supabase = get_supabase_client()
    if supabase is None:
        return False, "Supabase is not configured."

    try:
        response = supabase.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {"data": {"user_name": "namename"}},
            }
        )

        if response.user is not None and response.session is not None:
            st.session_state.user = response.user
            st.session_state.access_token = response.session.access_token
            st.session_state.refresh_token = response.session.refresh_token
            st.session_state.user_name = "namename"
            ensure_profile_exists()

        return True, "Account created. Check your email to confirm registration."
    except Exception as exc:
        return False, f"Registration error: {exc}"


def logout() -> None:
    """Clear authentication data from session state."""
    for key in AUTH_DEFAULTS:
        st.session_state[key] = None


def current_user() -> Any:
    """Return the currently authenticated user object or None."""
    return st.session_state.get("user")


def current_user_name() -> str:
    """Return the current username from session state."""
    return st.session_state.get("user_name", "namename")
