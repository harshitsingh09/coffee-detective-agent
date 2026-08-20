"""Streamlit interface for the Coffee Machine Detective."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from incident_assistant.application.errors import InvalidIncidentError
from incident_assistant.bootstrap import build_agent_investigation_service, ensure_demo_data
from incident_assistant.config import Settings
from incident_assistant.domain.agent_models import ToolExecutionStatus

SCENARIOS: dict[str, dict[str, Any]] = {
    "The Watery Espresso": {
        "machine_id": "CM-1001",
        "machine": "Lobby Latte",
        "difficulty": "Easy",
        "mystery": "The espresso looks more like coffee-flavoured water.",
        "incident": "CM-1001 is suddenly making watery espresso and finishing shots too quickly.",
        "clues": (
            "Bean sensor falls to 2%.",
            "Recent espresso cycles use unusually few grams of coffee.",
            "Logs report LOW_BEANS and WATERY_DRINK.",
        ),
        "expected_tools": ("get_supply_levels", "get_recent_brews", "search_application_logs"),
        "ground_truth": "Coffee bean hopper nearly empty",
        "fix": "Refill the hopper and run one calibration espresso.",
    },
    "The Foamless Cappuccino": {
        "machine_id": "CM-1002",
        "machine": "Breakroom Barista",
        "difficulty": "Easy",
        "mystery": "Cappuccino is being served with absolutely no foam drama.",
        "incident": "CM-1002 makes coffee, but every cappuccino arrives without milk foam.",
        "clues": (
            "Milk level reaches 0%.",
            "Recent cappuccino cycles have NO_FOAM status.",
            "Sensors report MILK_LINE_DISCONNECTED.",
        ),
        "expected_tools": ("get_sensor_alerts", "get_supply_levels", "get_recent_brews"),
        "ground_truth": "Milk line disconnected or milk supply empty",
        "fix": "Reconnect or refill the milk system, prime it, and test a cappuccino.",
    },
    "Hotter Than Office Gossip": {
        "machine_id": "CM-1003",
        "machine": "Executive Espresso",
        "difficulty": "Medium",
        "mystery": "The machine is alarmingly hot and aborting drinks.",
        "incident": "CM-1003 feels extremely hot and has started aborting brew cycles.",
        "clues": (
            "Boiler temperature climbs to 101 C.",
            "Recent brews abort with ABORTED_OVERHEAT.",
            "Logs report an OVERHEAT safety event.",
        ),
        "expected_tools": (
            "get_temperature_history",
            "get_sensor_alerts",
            "search_application_logs",
        ),
        "ground_truth": "Brewing system overheating",
        "fix": "Stop brewing, cool the unit, and inspect ventilation and thermostat control.",
    },
    "The Bitter Science Experiment": {
        "machine_id": "CM-1004",
        "machine": "Lab Cappuccino",
        "difficulty": "Medium",
        "mystery": "Every cup tastes as if the machine has developed resentment.",
        "incident": "Coffee from CM-1004 tastes unusually bitter and the machine looks dirty.",
        "clues": (
            "221 brew cycles have elapsed since cleaning.",
            "Recent brews carry BITTER_WARNING.",
            "Maintenance notes say descaling was postponed.",
        ),
        "expected_tools": ("get_cleaning_status", "get_recent_brews", "search_application_logs"),
        "ground_truth": "Cleaning cycle overdue",
        "fix": "Run the approved cleaning and descaling cycle, then make a fresh test drink.",
    },
    "The Innocent Machine": {
        "machine_id": "CM-1005",
        "machine": "Reception Roast",
        "difficulty": "Trick case",
        "mystery": "Someone says the coffee tastes odd, but the telemetry disagrees.",
        "incident": "Please check whether CM-1005 has any fault; one person disliked their coffee.",
        "clues": (
            "Supplies are comfortably above warning levels.",
            "Temperature and pressure are stable.",
            "No alerts or abnormal brew cycles are present.",
        ),
        "expected_tools": ("get_machine_health", "get_machine_status"),
        "ground_truth": "No machine fault detected",
        "fix": "Confirm the selected recipe and cup size before blaming the machine.",
    },
}

TOOL_PURPOSES = {
    "get_machine_status": "Identify the machine and inspect its latest controller state.",
    "get_recent_brews": "Look for failed, watery, foamless, bitter, or aborted drinks.",
    "get_supply_levels": "Check whether water, beans, or milk are running low.",
    "get_sensor_alerts": "Read explicit hardware and ingredient-sensor warnings.",
    "get_temperature_history": "Inspect boiler temperature and pressure trends.",
    "get_cleaning_status": "Check the cleaning counter and maintenance notes.",
    "search_application_logs": "Search bounded machine logs for supporting events.",
    "search_similar_incidents": "Compare the mystery with historical coffee cases.",
    "get_machine_health": "Collect a compact overall health summary.",
}

st.set_page_config(
    page_title="Coffee Machine Detective",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #fffaf3 0%, #f7efe4 100%); }
    [data-testid="stSidebar"] { background: #2f2118; color: #fff8ef; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.78); border: 1px solid #ddc7ad;
        border-radius: 14px; padding: 12px;
    }
    .detective-card {
        background: #fffdf9; border: 1px solid #d6b894; border-left: 6px solid #8b5e3c;
        border-radius: 14px; padding: 1rem 1.2rem; margin: .5rem 0 1rem 0;
        box-shadow: 0 5px 16px rgba(65, 42, 25, .08);
    }
    .clue-chip {
        display: inline-block; background: #efe0cf; color: #4b2f20; border-radius: 99px;
        padding: .25rem .65rem; margin: .15rem; font-size: .86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _runtime() -> tuple[Any, Settings, bool]:
    _apply_streamlit_secrets()
    settings = Settings.from_environment()
    initialized = ensure_demo_data(settings)
    return build_agent_investigation_service(settings), settings, initialized


def _apply_streamlit_secrets() -> None:
    supported = (
        "LLM_PROVIDER",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "ENABLE_AI_AGENT",
        "MAX_AGENT_STEPS",
        "MAX_LOG_RESULTS",
        "SIMILAR_INCIDENT_TOP_K",
        "SIMILARITY_THRESHOLD",
        "LLM_TIMEOUT_SECONDS",
    )
    try:
        for name in supported:
            if name in st.secrets and name not in os.environ:
                os.environ[name] = str(st.secrets[name])
    except FileNotFoundError:
        return


def _selected_scenario() -> dict[str, Any]:
    return SCENARIOS[st.session_state.scenario_selector]


def _choose_scenario() -> None:
    st.session_state.incident_text = _selected_scenario()["incident"]


def _render_scenario_dossier(scenario: dict[str, Any], *, reveal: bool = False) -> None:
    st.markdown(
        f"""
        <div class="detective-card">
          <div style="font-size:.8rem;letter-spacing:.08em;color:#8b5e3c">CASE FILE</div>
          <h3 style="margin:.2rem 0">{scenario["machine"]} · {scenario["machine_id"]}</h3>
          <p style="margin:.2rem 0"><b>Mystery:</b> {scenario["mystery"]}</p>
          <p style="margin:.2rem 0"><b>Difficulty:</b> {scenario["difficulty"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("View the synthetic clues used to build this scenario", expanded=reveal):
        for clue in scenario["clues"]:
            st.markdown(f"- 🔎 {clue}")
        st.markdown("**Tools that should be useful:**")
        st.code("\n".join(scenario["expected_tools"]), language="text")
    with st.expander("Reveal scenario ground truth", expanded=reveal):
        st.markdown(f"**Designed root cause:** {scenario['ground_truth']}")
        st.markdown(f"**Designed fix:** {scenario['fix']}")
        st.caption(
            "This is synthetic scenario metadata for learning and evaluation; it is not sent "
            "to the LLM as evidence."
        )


def _render_diagnosis(report: Any) -> None:
    diagnosis = report.diagnosis
    st.subheader("Detective's conclusion")
    status, confidence, mode, tools = st.columns(4)
    status.metric("Verdict", diagnosis.status.value.replace("_", " ").title())
    confidence.metric("Confidence", f"{diagnosis.confidence:.0%}")
    mode.metric("Execution", report.execution_mode.value.replace("_", " ").title())
    tools.metric("Clues checked", len(report.tool_results))
    st.markdown(f"### ☕ {diagnosis.root_cause or 'Mystery remains unsolved'}")
    st.write(diagnosis.explanation)

    evidence_column, action_column = st.columns(2)
    with evidence_column:
        st.markdown("#### Evidence pinned to the board")
        if diagnosis.supporting_evidence:
            for item in diagnosis.supporting_evidence:
                st.markdown(f"- {item}")
        else:
            st.caption("No conclusive evidence was recorded.")
    with action_column:
        st.markdown("#### Recommended next moves")
        for action in diagnosis.recommended_actions:
            st.markdown(f"- {action}")

    if diagnosis.warnings:
        with st.expander("Warnings and fallback notes", expanded=True):
            for warning in diagnosis.warnings:
                st.warning(warning)


def _render_detective_notebook(report: Any) -> None:
    st.subheader("Detective notebook — observable investigation")
    st.info(
        "You can see every safe action, argument, tool result, and cited fact. Private model "
        "chain-of-thought is not requested or displayed; the notebook is the auditable record."
    )

    st.markdown("#### Timeline")
    for event in report.trace:
        with st.container(border=True):
            event_name = event.event_type.value.replace("_", " ").title()
            st.markdown(f"**Step {event.sequence} · {event_name}**")
            st.write(event.message)
            if event.result_summary:
                st.caption(f"Observed result: {event.result_summary}")
            if event.duration_ms is not None:
                st.caption(f"Duration: {event.duration_ms:.1f} ms")

    if report.tool_results:
        st.markdown("#### Every tool call and returned clue")
        for index, result in enumerate(report.tool_results, start=1):
            succeeded = result.status is ToolExecutionStatus.SUCCEEDED
            marker = "✅" if succeeded else "⚠️"
            with st.expander(
                f"{marker} Clue {index}: {result.name} · {result.status.value}",
                expanded=True,
            ):
                purpose = TOOL_PURPOSES.get(result.name, "Evidence")
                st.markdown(f"**What this tool checks:** {purpose}")
                arguments, returned = st.columns(2)
                with arguments:
                    st.markdown("**Validated arguments**")
                    st.json(result.arguments)
                with returned:
                    st.markdown("**Returned evidence**")
                    st.json(result.data if result.data is not None else {"error": result.error})
                st.caption(f"Completed in {result.duration_ms:.1f} ms")

    if report.diagnosis.similar_incidents:
        st.markdown("#### Historical case matches")
        for match in report.diagnosis.similar_incidents:
            with st.expander(f"{match.incident_id} · {match.similarity_score:.0%} similar"):
                st.write(match.description)
                st.markdown(f"**Root cause:** {match.root_cause}")
                st.markdown(f"**Resolution:** {match.resolution}")
                st.caption(f"Retrieval method: {match.retrieval_method}")


def _render_scenario_lab() -> None:
    st.header("Scenario Lab")
    st.write(
        "These are all five deliberately seeded mysteries. Everything is synthetic, so you "
        "can inspect the designed clues and expected answer without exposing real operational data."
    )
    rows = [
        {
            "Case": name,
            "Machine": scenario["machine_id"],
            "Nickname": scenario["machine"],
            "Difficulty": scenario["difficulty"],
            "Designed root cause": scenario["ground_truth"],
        }
        for name, scenario in SCENARIOS.items()
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    for name, scenario in SCENARIOS.items():
        with st.expander(f"☕ {name} · {scenario['machine_id']}"):
            st.markdown(f"**Incident text:** {scenario['incident']}")
            st.markdown("**Seeded clues:**")
            for clue in scenario["clues"]:
                st.markdown(f"- {clue}")
            st.markdown(f"**Expected diagnosis:** {scenario['ground_truth']}")
            st.markdown(f"**Expected remedy:** {scenario['fix']}")
            tools = ", ".join(f"`{tool}`" for tool in scenario["expected_tools"])
            st.markdown("**Useful tools:** " + tools)


def _render_how_it_works(settings: Settings) -> None:
    st.header("How the detective works")
    st.code(
        """Coffee complaint containing CM-####
        ↓
Machine-ID extraction + safety guard
        ↓
Bounded provider-independent agent loop
        ↓
One of nine validated, read-only clue tools
        ↓
Structured telemetry / logs / similar cases
        ↓
Validated diagnosis + observable notebook
        ↓
Deterministic rules if AI is unavailable""",
        language="text",
    )
    st.subheader("The nine clue tools")
    st.dataframe(
        {
            "Allowlisted tool": list(TOOL_PURPOSES),
            "What it reveals": list(TOOL_PURPOSES.values()),
        },
        hide_index=True,
        width="stretch",
    )
    safe, transparent = st.columns(2)
    with safe:
        st.markdown("#### Safety rails")
        st.markdown(
            "- Read-only, parameterized SQL\n"
            "- Strict `CM-####` machine scoping\n"
            "- Maximum five agent steps by default\n"
            "- Repeated tool calls rejected\n"
            "- No arbitrary shell, Python, URLs, or writes"
        )
    with transparent:
        st.markdown("#### What the frontend reveals")
        st.markdown(
            "- Complete synthetic scenario setup\n"
            "- Every selected tool and validated argument\n"
            "- Every returned evidence object\n"
            "- Observable event timeline and durations\n"
            "- Evidence, recommendation, warnings, and fallback mode"
        )
        st.warning("Hidden chain-of-thought is intentionally not shown.")
    st.caption(
        f"Current provider configuration: {settings.llm_provider.value} · "
        f"{settings.selected_model or 'deterministic rules'}"
    )


service, settings, initialized = _runtime()

with st.sidebar:
    st.title("☕ Coffee Detective")
    st.caption("Solving office coffee crimes with telemetry, logs, and bounded AI")
    st.divider()
    ai_active = settings.ai_provider_configured
    st.markdown(f"**Mode:** {'AI detective' if ai_active else 'Rules detective'}")
    st.markdown(f"**Provider:** `{settings.llm_provider.value}`")
    st.markdown(f"**Model:** `{settings.selected_model or 'deterministic-rules'}`")
    st.markdown(f"**Clue limit:** {settings.max_agent_steps}")
    st.markdown("**Dataset:** 20 fictional machines · 10,000 synthetic brews")
    if initialized:
        st.success("Fresh coffee-machine evidence was seeded.")
    if settings.enable_ai_agent and not ai_active:
        st.info("Provider key unavailable; deterministic detective is active.")
    st.divider()
    st.caption("No real employees, customers, or coffee machines are represented.")

st.title("☕ Coffee Machine Detective")
st.caption("A playful, evidence-first AI agent for suspicious office coffee")

investigate_tab, scenarios_tab, architecture_tab = st.tabs(
    ["Investigate", "Scenario Lab", "How It Works"]
)

with investigate_tab:
    if "scenario_selector" not in st.session_state:
        st.session_state.scenario_selector = next(iter(SCENARIOS))
    if "incident_text" not in st.session_state:
        st.session_state.incident_text = _selected_scenario()["incident"]
    st.selectbox(
        "Choose a mystery",
        tuple(SCENARIOS),
        key="scenario_selector",
        on_change=_choose_scenario,
    )
    current_scenario = _selected_scenario()
    _render_scenario_dossier(current_scenario)
    incident = st.text_area(
        "Coffee complaint",
        key="incident_text",
        height=100,
        help="Include a fictional machine ID such as CM-1001. You may edit the sample.",
    )
    if st.button("Investigate this coffee crime", type="primary", width="stretch"):
        try:
            with st.spinner("Dusting the espresso machine for digital fingerprints..."):
                result = service.investigate(incident)
        except InvalidIncidentError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Investigation failed safely: {type(exc).__name__}: {exc}")
        else:
            _render_diagnosis(result)
            _render_detective_notebook(result)
            st.divider()
            st.subheader("Compare with the designed scenario")
            _render_scenario_dossier(current_scenario, reveal=True)

with scenarios_tab:
    _render_scenario_lab()

with architecture_tab:
    _render_how_it_works(settings)
