# Streamlit Seattle Weather dashboard

Just an example Streamlit dashboard exploring the classic Seattle Weather dataset.

## View it in one click

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://demo-seattle-weather.streamlit.app/)

## Try it on your machine

1. Get the code:

   ```sh
   $ git clone https://github.com/streamlit/demo-seattle-weather.git
   ```

2. Start a virtual environment and get the dependencies (requires uv):

   ```sh
   $ uv venv

   $ .venv/bin/activate

   $ uv sync
   ```

3. Start the app:

    ```sh
    $ streamlit run streamlit_app.py
    ```

## Streamlit Cloud configuration

To enable login, saved scenarios, and history in Streamlit Cloud, set these app secrets:

```toml
SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_ANON_OR_SERVICE_KEY"
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"
```

In Streamlit Cloud, open your app, then go to `Settings` -> `Secrets` and paste the values above.

The app also accepts the same values as environment variables when running outside Streamlit Cloud.
