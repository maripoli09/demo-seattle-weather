from typing import Any

import streamlit as st

from supabase_http import is_supabase_available, sign_in_with_password, sign_up, upsert_profile


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
    return None


def ensure_profile_exists() -> tuple[bool, str]:
    """Upsert the authenticated user profile in the profiles table."""
    user = st.session_state.get("user")
    if user is None:
        return False, "Invalid session."

    access_token = st.session_state.get("access_token")
    if not access_token or not is_supabase_available():
        return False, "Invalid session or Supabase is not configured."

    try:
        user_name = st.session_state.get("user_name") or "namename"
        upsert_profile({"id": user.id, "user_name": user_name}, access_token)
        return True, ""
    except Exception as exc:
        return False, f"Error ensuring profile: {exc}"


def login(email: str, password: str) -> tuple[bool, str]:
    """Authenticate a user and store tokens in session state."""
    if not email or not password:
        return False, "Email and password are required."

    if not is_supabase_available():
        return False, "Supabase is not configured."

    try:
        user, session = sign_in_with_password(email, password)

        if user is None or session is None:
            return False, "Authentication failed."

        st.session_state.user = user
        st.session_state.access_token = session.access_token
        st.session_state.refresh_token = session.refresh_token
        st.session_state.user_name = (
            getattr(user, "user_metadata", {}) or {}
        ).get("user_name", "namename")

        ok_profile, profile_msg = ensure_profile_exists()
        if not ok_profile:
            return False, profile_msg

        return True, f"Welcome, {user.email}!"
    except Exception as exc:
        return False, f"Login error: {exc}"


def register(email: str, password: str) -> tuple[bool, str]:
    """Create a user account and store tokens when a session is returned."""
    if not email or not password:
        return False, "Email and password are required."

    if not is_supabase_available():
        return False, "Supabase is not configured."

    try:
        user, session = sign_up(email, password, "namename")

        if user is not None and session is not None:
            st.session_state.user = user
            st.session_state.access_token = session.access_token
            st.session_state.refresh_token = session.refresh_token
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
