# Coffee Machine Detective

A playful, portfolio-ready AI agent that investigates fictional smart coffee-machine
mysteries using SQLite telemetry, bounded log search, optional local semantic retrieval,
and a provider-independent LLM loop.

The project is completely synthetic: the machines, brew cycles, sensor readings,
maintenance notes, logs, and incidents do not represent real people or equipment.

## What it does

A user enters a complaint such as:

> CM-1001 is suddenly making watery espresso and finishing shots too quickly.

The detective then:

```text
Extracts the CM-#### machine ID
        ↓
Chooses one safe clue tool
        ↓
Receives structured telemetry or logs
        ↓
Chooses another useful clue or stops
        ↓
Returns a validated diagnosis, evidence, and next actions
```

The application works without an API key through deterministic rules. With a configured
provider, Gemini, Groq, or OpenAI can select tools inside the same bounded controller.

## Five seeded mysteries

The seed creates 20 fictional machines and 10,000 synthetic brew cycles.

| Machine | Mystery | Designed root cause |
| --- | --- | --- |
| `CM-1001` | Watery, unusually fast espresso | Coffee bean hopper nearly empty |
| `CM-1002` | Cappuccino has no milk foam | Milk line disconnected or milk empty |
| `CM-1003` | Machine is hot and aborts drinks | Brewing system overheating |
| `CM-1004` | Coffee tastes unusually bitter | Cleaning cycle overdue |
| `CM-1005` | One subjective complaint, healthy telemetry | No machine fault detected |

The Streamlit **Scenario Lab** exposes every designed clue, expected useful tool, ground
truth, and remedy. Ground-truth metadata is for learning and evaluation; it is never sent
to the model as evidence.

## Nine safe clue tools

1. `get_machine_status(machine_id)`
2. `get_recent_brews(machine_id)`
3. `get_supply_levels(machine_id)`
4. `get_sensor_alerts(machine_id)`
5. `get_temperature_history(machine_id)`
6. `get_cleaning_status(machine_id)`
7. `search_application_logs(machine_id, keywords)`
8. `search_similar_incidents(description, top_k)`
9. `get_machine_health(machine_id)`

Every tool is read-only, allowlisted, bounded, and validated with Pydantic. There is no
arbitrary SQL, Python, shell, URL, filesystem-write, or remediation tool.

## Transparent frontend

The frontend contains three views:

- **Investigate** — select or edit a mystery, run it, and see the diagnosis.
- **Scenario Lab** — inspect all scenarios, clues, intended tools, and answers.
- **How It Works** — see the architecture, tool surface, and safety controls.

After an investigation, the **Detective Notebook** shows:

- every observable controller event;
- every selected tool;
- every validated argument;
- every structured returned result;
- timings, failures, and fallback warnings;
- cited evidence and recommended actions;
- similar historical cases when that optional tool was selected.

The application does not request or reveal private model chain-of-thought. The observable
notebook is the auditable reasoning record: actions, evidence, and conclusions rather than
hidden token-by-token deliberation.

## Architecture

```text
Streamlit / CLI
      ↓
AgentInvestigationService
      ├── machine-ID and prompt-injection guard
      ├── maximum step budget
      ├── duplicate-call rejection
      ├── machine-scope validation
      └── deterministic fallback
      ↓
AgentLLMProvider factory
      ├── Gemini Interactions API
      ├── Groq Chat Completions
      ├── OpenAI Responses API
      └── rules / no provider
      ↓
Nine read-only coffee clue tools
      ├── SQLite telemetry and maintenance data
      ├── bounded text logs
      └── optional local historical retrieval
      ↓
IncidentDiagnosis + observable trace
```

The domain and application layers contain no vendor SDK logic. Provider adapters translate
vendor calls into the same `AgentStep`, `RequestedToolCall`, and `IncidentDiagnosis`
objects.

## Data model

SQLite contains:

- `machines` — nickname, office location, model, and active state;
- `brew_cycles` — drink type, outcome, duration, ingredient use, and temperature;
- `sensor_readings` — water, beans, milk, temperature, pressure, cleaning counter, alerts;
- `maintenance_events` — cleaning date, note, and resolution state;
- `incidents` — synthetic historical symptoms, root causes, and remedies.

`data/app_logs.txt` contains scoped machine events. `data/incidents.csv` is the transparent
historical-case export.

## Provider configuration

Copy `.env.example` to `.env`. `.env` is ignored by Git.

Groq example:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b
ENABLE_AI_AGENT=true
```

Other choices:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash-lite
```

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.4-mini
```

Guaranteed offline mode:

```dotenv
LLM_PROVIDER=rules
ENABLE_AI_AGENT=true
```

Cost and safety controls:

```dotenv
MAX_AGENT_STEPS=5
MAX_LOG_RESULTS=20
SIMILAR_INCIDENT_TOP_K=3
SIMILARITY_THRESHOLD=0.35
LLM_TIMEOUT_SECONDS=30
```

The similarity threshold is an initial configurable value, not an optimized claim.

## Local setup

Python 3.11 or newer is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe seed_data.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`.

For optional local semantic incident retrieval, install the larger embedding stack:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
```

The standard deployment uses the lightweight lexical history fallback, avoiding the
large PyTorch download required by Sentence Transformers.

CLI example:

```powershell
.\.venv\Scripts\python.exe -m incident_assistant "CM-1001 is making watery espresso"
```

Provider-aware smoke test:

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_test "CM-1001 is making watery espresso"
```

Add `--rules` to verify the local deterministic fallback without using API credits.

## Tests and quality checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
```

The suite covers all five diagnoses, all nine tools, schemas, machine scoping, provider
translation, identical cross-provider orchestration, fallback behavior, timeouts, rate
limits, malformed calls and diagnoses, prompt injection, local retrieval, Streamlit
rendering, and data generation. Provider tests are mocked and consume no API credits.

## Deploy on Streamlit Community Cloud

1. Push this directory to its own GitHub repository.
2. At `share.streamlit.io`, create an app from the repository and select `app.py`.
3. Select Python 3.11 in **Advanced settings**.
4. Paste the following into the **Secrets** field, replacing only the placeholder key:

```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "paste-your-groq-key-here"
GROQ_MODEL = "openai/gpt-oss-20b"
ENABLE_AI_AGENT = "true"
```

The local `.env` and `.streamlit/secrets.toml` are ignored by Git. Never commit either
file. Community Cloud injects the values securely at runtime, and `app.py` maps those
secret values into the application's environment-based configuration.

## Project layout

```text
.
├── app.py
├── seed_data.py
├── incident_assistant/
│   ├── application/       # deterministic and agent controllers
│   ├── domain/            # strict models and provider-neutral ports
│   ├── infrastructure/    # coffee data, logs, retrieval, provider adapters
│   ├── tools/             # nine allowlisted clue tools
│   └── bootstrap.py
├── data/                  # generated synthetic coffee evidence
├── scripts/               # index and evaluation utilities
├── tests/
├── Knowledge Base/
├── requirements.txt
└── .env.example
```

## Resume-ready summary

> **Coffee Machine Detective | Python, SQL, LLM Tool Use, Streamlit**
> Built a provider-independent AI investigation agent that diagnoses synthetic smart
> coffee-machine faults using nine validated telemetry, log, maintenance, and historical-
> retrieval tools. Implemented bounded tool orchestration, structured diagnoses,
> observable evidence traces, local semantic retrieval, and deterministic no-key fallback
> across Gemini, Groq, and optional OpenAI providers.

## Limitations and future ideas

- The five faults are deliberately seeded and should not be presented as real predictive
  maintenance performance.
- The local historical corpus contains only five incidents.
- The agent recommends actions but never controls a physical machine.
- A future version could add synthetic energy usage, drink-rating feedback, or a visual
  machine floor map after those additions have measurable value.
