# Coffee Machine Detective Knowledge Base

## Project Goal

Coffee Machine Detective is a synthetic, tool-using AI application that investigates
fictional smart coffee-machine complaints. It demonstrates bounded agent orchestration,
structured tool use, SQL evidence, log analysis, local historical retrieval, provider
independence, and deterministic fallback through an approachable theme.

## Investigation Flow

```text
Coffee complaint containing CM-####
        ↓
Machine ID extraction and safety validation
        ↓
Gemini, Groq, OpenAI, or deterministic rules
        ↓
One relevant allowlisted clue tool at a time
        ↓
Structured telemetry, log, maintenance, or historical evidence
        ↓
Validated diagnosis, evidence, recommendations, and observable notebook
```

## Synthetic Scenarios

| Machine | Symptom | Root cause | Primary evidence |
| --- | --- | --- | --- |
| `CM-1001` | Watery, fast espresso | Bean hopper nearly empty | 2% beans, low dose, `LOW_BEANS` logs |
| `CM-1002` | Cappuccino has no foam | Milk line disconnected or empty | 0% milk, `NO_FOAM`, milk-line alert |
| `CM-1003` | Hot machine aborts drinks | Brewing system overheating | 101 C, aborted brews, `OVERHEAT` logs |
| `CM-1004` | Coffee tastes bitter | Cleaning cycle overdue | 221 cycles, postponed cleaning, residue warning |
| `CM-1005` | Subjective complaint only | No machine fault detected | Healthy supplies, temperature, pressure, and brews |

The seed contains 20 fictional machines and 10,000 brew cycles. Only the five machines
above have deliberate scenario behavior; the remaining machines are healthy references.

## Database Tables

- `machines`: identity, nickname, location, model, and active state.
- `brew_cycles`: drink type, status, duration, ingredient use, and temperature.
- `sensor_readings`: water, bean, and milk levels plus temperature, pressure, cleaning
  counter, and alerts.
- `maintenance_events`: cleaning records and technician notes.
- `incidents`: historical synthetic symptoms, causes, and remedies.

## Agent Tools

1. `get_machine_status()`
2. `get_recent_brews()`
3. `get_supply_levels()`
4. `get_sensor_alerts()`
5. `get_temperature_history()`
6. `get_cleaning_status()`
7. `search_application_logs()`
8. `search_similar_incidents()`
9. `get_machine_health()`

All tools are read-only, bounded, and schema-validated. The agent cannot execute arbitrary
SQL, Python, shell commands, URLs, filesystem operations, or physical remediation.

## Frontend Transparency

The Streamlit application exposes:

- every scenario's complaint, designed clues, expected useful tools, ground truth, and fix;
- the actual execution mode and provider;
- every observable controller event;
- selected tools and validated arguments;
- complete structured tool results and timings;
- final supporting evidence, recommended actions, warnings, and similar incidents.

Hidden model chain-of-thought is neither requested nor displayed. The observable notebook
provides the appropriate audit trail: actions, evidence, and conclusions.

## Provider and Fallback Design

The application depends on `AgentLLMProvider`, not a vendor SDK. The factory supports
Gemini, Groq, OpenAI, and rules. Missing keys, API failures, timeouts, rate limits,
malformed calls, invalid diagnoses, and step-limit exhaustion activate deterministic
fallback. Reports label the actual execution mode rather than implying AI was used.

## Success Criteria

The project is successful when it can:

1. Extract a valid `CM-####` identifier.
2. Select only useful coffee-machine tools rather than running a fixed pipeline.
3. Ground a diagnosis in returned evidence.
4. Diagnose all four seeded faults and the healthy control.
5. Remain runnable without any API key.
6. Show a complete observable investigation without exposing private reasoning.
