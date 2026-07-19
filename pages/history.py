import streamlit as st
import pandas as pd
from supabase import create_client, Client

st.set_page_config(page_title="Historico de Simulacoes", layout="wide")
st.title("Historico de Simulacoes")
st.caption("Ultimas simulacoes guardadas no Supabase")

@st.cache_resource
def get_supabase_client() -> Client | None:
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

def fetch_simulations(limit: int = 50):
    supabase = get_supabase_client()
    if supabase is None:
        return None, "Supabase nao configurado."

    try:
        res = (
            supabase
            .table("simulations")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        data = res.data if hasattr(res, "data") else []
        return data, None
    except Exception as e:
        return None, f"Erro ao ler historico: {e}"

limit = st.slider("Numero de simulacoes", min_value=10, max_value=200, value=50, step=10)

data, err = fetch_simulations(limit=limit)

if err:
    st.warning(err)
else:
    if not data:
        st.info("Ainda nao existem simulacoes guardadas.")
    else:
        df = pd.DataFrame(data)
        cols_preferidas = [
            "created_at",
            "city",
            "cycle",
            "price_type",
            "num_solar_panels",
            "panel_wattage",
            "predicted_consumption",
            "predicted_production",
            "energy_balance",
            "price_now",
            "estimated_cost_without_solar",
            "estimated_cost_with_solar",
            "estimated_savings",
            "model_version",
        ]
        cols_visiveis = [c for c in cols_preferidas if c in df.columns]
        st.dataframe(df[cols_visiveis], use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descarregar historico (CSV)",
            data=csv,
            file_name="historico_simulacoes.csv",
            mime="text/csv",
        )