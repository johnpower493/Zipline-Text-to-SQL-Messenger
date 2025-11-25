# CircularQuery - Release 1
## Your Local AI data analyst - inside Slack
<div style="display: flex; align-items: center; gap: 12px;">
  <img src="examples/logo-small.png" alt="CircularQuery logo" width="256">
  <p>
    Ask questions in Slack → get <strong>safe, read-only SQL</strong> (SQLite) via
    <strong>Ollama</strong> (IBM Granite Micro or any local model), plus
    <strong>paginated results</strong>, <strong>CSV export</strong>, and
    <strong>inline plots</strong> — no data leaves your machine.
  </p>
</div>


> Works with `/dd` (CircularQuery SQL assistant), CSV exports, and "Plot Data" interactivity.

---

## Demo (what it does)

- `/dd top 10 tracks by revenue last year`
  - Generates a **SQLite SELECT** using your schema
  - Executes against **Chinook** (or your DB)
  - Replies with a paginated table
  - Buttons: **Export CSV**, **📊 Bar Plot**, **📈 Line Plot** (choose X/Y, uploads chart), **🔍 Insights** (AI-powered analysis)

### Query examples:
![Query_step_3](examples/query3.gif)
### Simply use /dd 'your data question'
![Query step 1](examples/query1.png)
### Guard rails example:
![Guardrails example](examples/guardrails.png)
### Plotting xample:
![Plot example](examples/plot.png)

---

## Quickstart

### 1) Prereqs
- Python 3.10+  
- Slack workspace where you can install custom apps
- [Ollama](https://ollama.com/) running locally
- SQLite DB (Chinook recommended for demo)

### 2) Clone & install

`git clone https://github.com/johnpower493/CircularQuery.git` \
`cd CircularQuery` \
`python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate` \
`pip install -r requirements.txt` \

### 3) Pull a local model

Pick one you like (examples):
ollama pull  granite4
ollama pull qwen3:4b

### 4) Copy env and set tokens
cp .env.example .env
# then edit .env
```
.env keys

SLACK_BOT_TOKEN — from “Install App” → Bot User OAuth Token (xoxb-...)

OLLAMA_BASE_URL — default http://127.0.0.1:11434

OLLAMA_MODEL — e.g. qwen3:4b

SQLITE_PATH — ./chinook.db (or your DB)

Optional: SCHEMA_YAML_PATH if you want to override auto-introspection
```

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

Invite the bot/app to your channel: /invite @CircularQuery

Try: /dd top 5 customers by total spend
Use the Export CSV / 📊 Bar Plot / 📈 Line Plot / 🔍 Insights buttons.

### Architecture
```
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
```

### Security & Guardrails

Read-only DB (SQLite opened with mode=ro)

SQL allowlist: only SELECT (+ optional CTE WITH)

Auto-append LIMIT (defaults to 500 unless provided)

Simple timeout on LLM call; result size caps for CSV

### Config / Env

SLACK_BOT_TOKEN (required): bot token with chat:write, chat:write.public, files:write, commands

OLLAMA_MODEL: e.g. qwen3:4b

SQLITE_PATH: path to your DB, default ./chinook.db
