# Data Distillery Slack SQL Cockpit with Local LLM

Ask questions in Slack → get **safe, read-only SQL** (SQLite) via **Ollama** (IBM Granite Micro or any local model), plus **paginated results**, **CSV export**, and **inline plots** — no data leaves your machine.

> Works with `/dd` (Data Distillery SQL assistant), CSV exports, and “Plot Data” interactivity.

---

## Demo (what it does)

- `/dd top 10 tracks by revenue last year`
  - Generates a **SQLite SELECT** using your schema
  - Executes against **Chinook** (or your DB)
  - Replies with a paginated table
  - Buttons: **Export CSV**, **Plot Data** (choose X/Y, uploads chart)

---

## Quickstart

### 1) Prereqs
- Python 3.10+  
- Slack workspace where you can install custom apps
- [Ollama](https://ollama.com/) running locally
- SQLite DB (Chinook recommended for demo)

### 2) Clone & install

git clone https://github.com/johnpower493/Slack-SQL-Cockpit-Local-LLM.git
cd slack-sql-copilot
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### 3) Pull a local model

Pick one you like (examples):
ollama pull  granite4
ollama pull qwen3:4b

### 4) Copy env and set tokens
cp .env.example .env
# then edit .env

.env keys

SLACK_BOT_TOKEN — from “Install App” → Bot User OAuth Token (xoxb-...)

OLLAMA_BASE_URL — default http://127.0.0.1:11434

OLLAMA_MODEL — e.g. qwen3:4b

SQLITE_PATH — ./chinook.db (or your DB)

Optional: SCHEMA_YAML_PATH if you want to override auto-introspection

### 5) Slack app setup (one-shot manifest)

In api.slack.com/apps
 → Create New App → From an app manifest

Pick your workspace → paste the manifest from manifests/slack_app_manifest.yaml (below).

Install the app to your workspace.

In Interactivity & Shortcuts:

Turn Interactivity ON

Request URL: https://<your-ngrok-domain>/slack/interactions

In each Slash Command edit page, set the Request URL to your public URL paths:

/dd → https://<your-ngrok-domain>/slack/sqlquery

/help → https://<your-ngrok-domain>/slack/help

If you’re local, run ngrok (or Cloudflare tunnel):
ngrok http http://localhost:5000

### 6) Run the app
`python app.py`

### 7) In Slack

Invite the bot/app to your channel: /invite @DataDistillery

Try: /dd top 5 customers by total spend
Use the Export CSV / Plot Data buttons.

### Architecture
Slack Slash Commands  +  Interactivity (Buttons/Selects)
                | (HTTP Webhooks)
                v
           Flask server  <——>  Guardrails (SELECT-only, LIMIT, timeouts)
                |                     |
                |                 Prompting
                v                     v
           Ollama (Local LLM)  ——>  SQL (SQLite dialect)
                |
                v
          SQLite (read-only) → results → CSV / Plot → Slack files.upload

### Security & Guardrails

Read-only DB (SQLite opened with mode=ro)

SQL allowlist: only SELECT (+ optional CTE WITH)

Auto-append LIMIT (defaults to 500 unless provided)

Simple timeout on LLM call; result size caps for CSV

### Config / Env

SLACK_BOT_TOKEN (required): bot token with chat:write, chat:write.public, files:write, commands

OLLAMA_MODEL: e.g. qwen3:4b

SQLITE_PATH: path to your DB, default ./chinook.db
