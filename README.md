# Smart Energy Advisor

Aplicação Streamlit para apoio à gestão energética residencial, com previsão de consumo por XGBoost, estimativa de produção solar, comparação entre ciclos tarifários e histórico de simulações com autenticação via Supabase.

## O que o projeto faz

O Smart Energy Advisor combina três blocos principais:

- previsão de consumo energético com um modelo XGBoost treinado sobre dados agregados do dataset Ausgrid;
- estimativa de produção solar com base na configuração da instalação e nas condições meteorológicas obtidas via OpenWeatherMap;
- simulação de custos e poupança para ciclos tarifários portugueses simples, bi-horário e tri-horário.

Além da simulação principal, a aplicação inclui autenticação, histórico de cenários guardados e páginas de apoio para análise exploratória dos dados e explicação do modelo.

## Funcionalidades

- Resumo energético atual com consumo previsto, produção solar, balanço e preço atual.
- Previsão horária do dia com cruzamento entre meteorologia e consumo previsto.
- Simulador de custos para as próximas 24 horas com e sem autoconsumo solar.
- Comparação visual entre os três ciclos tarifários.
- Recomendações contextuais baseadas em preço, produção solar e nebulosidade.
- Histórico de simulações por utilizador com filtros e exportação CSV.
- Página de análise do modelo e página de exploração dos dados agregados.

## Estrutura do repositório

```text
.
├── streamlit_app.py         # App principal e fluxo da interface
├── utils.py                 # Carregamento de modelos, previsão e recomendações
├── tariffs.py               # Lógica tarifária simplificada
├── weather.py               # Integração com OpenWeatherMap
├── supabase_http.py         # Autenticação e persistência via HTTP/Supabase
├── config.py                # Leitura centralizada de segredos e variáveis de ambiente
├── auth.py                  # Utilitários auxiliares de autenticação
├── pages/
│   ├── dashboard.py         # Exploração visual do dataset agregado
│   ├── history.py           # Histórico de simulações guardadas
│   └── AI_model.py          # Explicação e métricas do modelo
├── notebook.ipynb           # Notebook de exploração, preparação e treino
├── smart_energy_model.pkl   # Modelo XGBoost treinado
├── scaler.pkl               # Scaler utilizado no pipeline de inferência
├── df_gc_clean.pkl          # Dataset agregado e preparado para modelação
├── pyproject.toml
└── requirements.txt
```

## Requisitos

- Python 3.11 ou superior
- Conta e chave da API OpenWeatherMap
- Projeto Supabase, se quiseres ativar login, histórico e persistência

## Instalação

### Opção 1: com `uv`

```sh
uv venv
source .venv/bin/activate
uv sync
```

### Opção 2: com `pip`

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuração

Podes configurar as credenciais através de variáveis de ambiente ou de segredos do Streamlit.

### Variáveis de ambiente

```sh
export OPENWEATHER_API_KEY="a_tua_chave"
export SUPABASE_URL="https://teu-projeto.supabase.co"
export SUPABASE_KEY="a_tua_supabase_key"
```

### `.streamlit/secrets.toml`

```toml
OPENWEATHER_API_KEY = "a_tua_chave"
SUPABASE_URL = "https://teu-projeto.supabase.co"
SUPABASE_KEY = "a_tua_supabase_key"
```

Também são aceites aliases de configuração definidos em [config.py](/workspaces/demo-seattle-weather/config.py).

## Executar a aplicação

```sh
streamlit run streamlit_app.py
```

Depois abre o endereço mostrado no terminal, normalmente `http://localhost:8501`.

## Executar o notebook

O notebook [notebook.ipynb](/workspaces/demo-seattle-weather/notebook.ipynb) reúne a preparação dos dados, análise exploratória, treino do modelo, avaliação e exportação dos artefactos `.pkl`.

Se quiseres correr o notebook localmente, instala também as dependências de desenvolvimento:

```sh
pip install jupyter notebook ipykernel matplotlib seaborn
```

## Artefactos do modelo

- [smart_energy_model.pkl](/workspaces/demo-seattle-weather/smart_energy_model.pkl): modelo XGBoost treinado.
- [scaler.pkl](/workspaces/demo-seattle-weather/scaler.pkl): `StandardScaler` guardado para consistência do pipeline.
- [df_gc_clean.pkl](/workspaces/demo-seattle-weather/df_gc_clean.pkl): dataset agregado e preparado para inferência e análise.

## Notas sobre a implementação

- O modelo de consumo trabalha com features temporais e lags históricos.
- A estimativa solar usa uma aproximação simplificada baseada em potência instalada, hora, mês, temperatura e nebulosidade.
- A lógica tarifária atual é simplificada e serve como simulador operacional do protótipo.
- Se o Supabase não estiver configurado, a app continua funcional, mas o login, o histórico e o guardar de cenários ficam desativados.

## Dependências principais

- Streamlit
- Pandas
- NumPy
- Plotly
- Requests
- XGBoost
- scikit-learn
- Supabase
- Joblib

## Licença

Este projeto inclui a licença disponível em [LICENSE](/workspaces/demo-seattle-weather/LICENSE).
