import os

import re
import io
import json
import time
import yaml
import hashlib
import threading
import requests
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from threading import Thread
from core.guardrails import sanitize_sql

# ============
# ENV & CONFIG
# ============
load_dotenv()
SLACK_BOT_TOKEN   = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_API_BASE    = "https://slack.com/api"
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "granite4")
SQLITE_PATH       = os.getenv("SQLITE_PATH", "./chinook.db")
SCHEMA_YAML_PATH  = os.getenv("SCHEMA_YAML_PATH", "./schema.yaml")
NGROK_AUTHTOKEN   = os.getenv("NGROK_AUTHTOKEN", "")



SQLITE_PATH = os.environ.get("SQLITE_PATH", "./chinook.db")  # point to your local Chinook DB

# Optional: keep schema.yaml support — otherwise we’ll auto-infer from sqlite
SCHEMA_YAML_PATH = os.environ.get("SCHEMA_YAML_PATH", "./schema.yaml")

# Local storage for transient exports
EXPORTS_DIR = "./exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

app = Flask(__name__)

# ==================
# SCHEMA ENUMERATION
# ==================
def load_schema_from_yaml(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        try:
            cfg = yaml.safe_load(f)
            return cfg.get("database_schema")
        except Exception:
            return None

def load_schema_from_sqlite(db_path):
    """
    Build a compact, LLM-friendly schema string from SQLite PRAGMAs.
    Example (Chinook):
      - Albums(AlbumId INTEGER PK, Title TEXT, ArtistId INTEGER FK->Artists.ArtistId)
      - Tracks(TrackId INTEGER PK, Name TEXT, AlbumId INTEGER FK->Albums.AlbumId, ...)
    """
    if not os.path.exists(db_path):
        return "Database file not found."

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # list tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    tables = [r["name"] for r in cur.fetchall()]

    lines = []
    for t in tables:
        # columns
        cur.execute(f"PRAGMA table_info('{t}')")
        cols = cur.fetchall()
        col_parts = []
        pks = [c["name"] for c in cols if c["pk"]]
        for c in cols:
            name = c["name"]
            ctype = c["type"]
            pk = " PK" if c["pk"] else ""
            col_parts.append(f"{name} {ctype}{pk}")

        # foreign keys (best-effort)
        cur.execute(f"PRAGMA foreign_key_list('{t}')")
        fks = cur.fetchall()
        fk_parts = []
        for fk in fks:
            # fk['table'] -> parent table, fk['from'] -> column, fk['to'] -> parent col
            fk_parts.append(f"{fk['from']} -> {fk['table']}.{fk['to']}")

        line = f"- {t}(" + ", ".join(col_parts) + ")"
        if fk_parts:
            line += " FKs[" + "; ".join(fk_parts) + "]"
        lines.append(line)

    con.close()
    return "\n".join(lines)

DATABASE_SCHEMA = load_schema_from_yaml(SCHEMA_YAML_PATH) or load_schema_from_sqlite(SQLITE_PATH)

# ===============
# OLLAMA LLM
# ===============
def generate_prompt(question, schema):
    return (
        "You are a strict SQLite SQL generator. "
        "Always respond with ONLY a valid SQLite SELECT statement (no explanations, no markdown fences). "
        "Use the provided schema. Avoid DDL/DML; only read data. "
        f"\n\nSCHEMA:\n{schema}\n\n"
        f"Question: {question}\n"
        "SQL:"
    )

def get_sql_query(question, schema):
    """
    Use Ollama /api/chat to produce a single SELECT statement for SQLite.
    """
    prompt = generate_prompt(question, schema)

    try:
        # ollama chat API
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False
            },
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()

        # ollama returns {"message":{"role":"assistant","content":"..."}, ...}
        content = (data.get("message") or {}).get("content", "") or ""
        # Strip fences if present, keep only SQL
        content = content.strip().strip("`")
        content = re.sub(r"^sql\n", "", content, flags=re.IGNORECASE).strip()

        # safety: only allow SELECT (and optional WITH ... SELECT)
        if not re.match(r"^\s*(WITH\s+.+?\s+)?SELECT\b", content, re.IGNORECASE | re.DOTALL):
            return None

        # SQLite ends don't need semicolon for execution; drop if present
        content = content.rstrip(";").strip()
        ok, safe_sql, reason = sanitize_sql(content, default_limit=500)
        if not ok:
            print("Guardrails rejected SQL:", reason, content)
            return None
        return safe_sql        
    except Exception as e:
        print("Ollama error:", e)
        return None

# ===================
# SQLITE EXECUTION
# ===================
def execute_sql_query(sql_query):
    """
    Return JSON string or {"error": "..."}.
    """
    try:
        # con = sqlite3.connect(SQLITE_PATH)
        con = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        result = [dict(r) for r in rows]
        con.close()
        return json.dumps(result, default=str)
    except Exception as e:
        return {"error": str(e)}

# ====================
# FORMATTING UTILITIES
# ====================
def json_to_table(json_data, max_width=30):
    try:
        df = pd.read_json(io.StringIO(json_data))
        if df.empty:
            return "```\n<no rows>\n```"

        def trunc(s, L=max_width):
            s = str(s)
            return s if len(s) <= L else s[:L-3] + "..."

        df = df.applymap(trunc)  # <- use applymap, not map
        max_lengths = {col: min(max(df[col].astype(str).apply(len).max(), len(col)), max_width) for col in df.columns}

        result = "```\n"
        header = " | ".join(f"{col:{max_lengths[col]}}" for col in df.columns)
        sep = "-|-".join("-" * max_lengths[col] for col in df.columns)
        result += header + "\n" + sep + "\n"
        for _, row in df.iterrows():
            safe_cells = [str(c).replace("\n", " ").replace("|", "/") for c in row]
            result += " | ".join(f"{cell:{max_lengths[col]}}" for col, cell in zip(df.columns, safe_cells)) + "\n"
        result += "```"
        return result
    except Exception as e:
        return "Failed to format table: " + str(e)

def paginate_sql_query(sql_query, page_number, rows_per_page=12):
    # If the original already has LIMIT/OFFSET, we’ll trust it; otherwise append
    if re.search(r"\bLIMIT\b", sql_query, re.IGNORECASE):
        return sql_query
    offset = max(0, (page_number - 1) * rows_per_page)
    return f"{sql_query} LIMIT {rows_per_page} OFFSET {offset}"

# ==============================
# SLACK HELPERS (files + message)
# ==============================
def slack_post_message(channel_id, text=None, blocks=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}"
    }
    payload = {"channel": channel_id}
    if text:
        payload["text"] = text
    if blocks:
        payload["blocks"] = blocks
    r = requests.post(f"{SLACK_API_BASE}/chat.postMessage", headers=headers, json=payload)
    if not r.ok:
        print("Slack chat.postMessage error:", r.text)
    return r.ok, r.json() if r.ok else r.text

def slack_upload_file(ch_id, file_bytes, filename, title=None, initial_comment=None):
    """
    Upload a file to Slack using the new external upload flow:
      1) files.getUploadURLExternal
      2) POST bytes to the returned upload_url
      3) files.completeUploadExternal

    Returns (ok: bool, resp: dict|str)
    """
    if not SLACK_BOT_TOKEN:
        err = "SLACK_BOT_TOKEN is not set"
        print("Slack upload error:", err)
        return False, err

    # ---------- 1) Get upload URL + file_id ----------
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}"
    }

    length = len(file_bytes)

    # Slack accepts token either as header OR form field.
    # We'll send both to be extra-safe.
    data = {
        "filename": filename,
        "length": length,
        "token": SLACK_BOT_TOKEN,
    }

    r1 = requests.post(
        f"{SLACK_API_BASE}/files.getUploadURLExternal",
        headers=headers,
        data=data,
        timeout=30,
    )

    if not r1.ok:
        print("files.getUploadURLExternal HTTP error:", r1.status_code, r1.text)
        return False, r1.text

    j1 = r1.json()
    if not j1.get("ok"):
        print("files.getUploadURLExternal Slack error:", j1)
        return False, j1

    upload_url = j1["upload_url"]
    file_id    = j1["file_id"]

    # ---------- 2) Upload bytes to upload_url ----------
    # Docs: can send raw bytes or multipart; raw bytes is simplest. :contentReference[oaicite:1]{index=1}
    r2 = requests.post(
        upload_url,
        data=file_bytes,
        headers={"Content-Type": "application/octet-stream"},
        timeout=60,
    )

    if not r2.ok:
        print("Upload to upload_url HTTP error:", r2.status_code, r2.text)
        return False, r2.text

    # ---------- 3) Complete upload + share to channel ----------
    complete_payload = {
        "files": [
            {
                "id": file_id,
                "title": title or filename,
            }
        ],
        "channel_id": ch_id,
    }
    if initial_comment:
        complete_payload["initial_comment"] = initial_comment

    r3 = requests.post(
        f"{SLACK_API_BASE}/files.completeUploadExternal",
        headers={
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=complete_payload,
        timeout=30,
    )

    if not r3.ok:
        print("files.completeUploadExternal HTTP error:", r3.status_code, r3.text)
        return False, r3.text

    j3 = r3.json()
    if not j3.get("ok"):
        print("files.completeUploadExternal Slack error:", j3)
        return False, j3

    # Success 🎉
    return True, j3



# Keep in-memory selections per user
axis_selections = {}

def store_axis_selection(user_id, axis_type, selection):
    axis_selections.setdefault(user_id, {})[axis_type] = selection

def fetch_axis_selection(user_id):
    s = axis_selections.get(user_id, {})
    return s.get("X"), s.get("Y"), s.get("CSV")

def extract_sql_from_original_message(payload):
    """
    Pull the SQL back out of the original Slack message text:
    e.g. "SQL Query (SQLite): ```SELECT ...``` ..."
    """
    original = (payload.get("original_message") or {}).get("text") or ""
    m = re.search(r"```(.*?)```", original, re.DOTALL)
    if not m:
        return None

    sql = m.group(1).strip()
    # Strip optional leading "sql\n" and trailing semicolon
    sql = re.sub(r"^sql\n", "", sql, flags=re.IGNORECASE).strip()
    sql = sql.rstrip(";").strip()
    return sql or None


# ================
# ROUTE: downloads
# ================
@app.route('/exports/<filename>')
def download_csv(filename):
    # Only for local dev; Slack won’t fetch localhost anyway. We now prefer files.upload.
    return send_from_directory(EXPORTS_DIR, filename)

# =================
# ROUTE: /slack/sqlquery
# =================
def make_button(name, text, value):
    return {
        "name": name,
        "text": text,
        "type": "button",
        "value": value,
        "style": "primary"
    }

@app.route('/slack/sqlquery', methods=['POST'])
def handle_query():
    data = request.form
    response_url = data.get('response_url')
    command_text = (data.get('text') or "").strip()
    channel_id = data.get("channel_id")

    # Extract trailing <page>
    page_match = re.search(r"<(\d+)>$", command_text)
    if page_match:
        page_number = int(page_match.group(1))
        query_text = re.sub(r"\s*<\d+>$", "", command_text).strip()
    else:
        page_number = 1
        query_text = command_text

    def work():
        rows_per_page = 12
        sql_query = get_sql_query(query_text, DATABASE_SCHEMA)
        if not sql_query:
            requests.post(response_url, json={"response_type": "in_channel",
                                              "text": "I couldn't generate a safe SELECT for SQLite. Try rephrasing."})
            return

        # Run once to get full result (for page count)
        result = execute_sql_query(sql_query)
        if isinstance(result, dict) and "error" in result:
            requests.post(response_url, json={"response_type": "in_channel",
                                              "text": f"SQL error: {result['error']}"})
            return

        result_json = json.loads(result)
        total_rows = len(result_json)
        if total_rows == 0:
            requests.post(response_url, json={"response_type": "in_channel",
                                              "text": f"SQL Query: ```{sql_query}```\nNo data."})
            return

        total_pages = (total_rows + rows_per_page - 1) // rows_per_page
        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        page_slice = result_json[start_idx:end_idx]
        formatted = json_to_table(json.dumps(page_slice))

        # Buttons
        buttons = [
            make_button("export_csv", "Export as CSV", query_text),
            make_button("plot", "Bar Plot", query_text),
        ]
        if total_pages > 1:
            prev_page = max(1, page_number - 1)
            next_page = min(total_pages, page_number + 1)
            buttons.append(make_button("previous", f"Previous ({prev_page})", f"{prev_page}|{query_text}"))
            buttons.append(make_button("next", f"Next ({next_page})", f"{next_page}|{query_text}"))

        attachments = [{
            "text": "Navigate or export:",
            "fallback": "Actions not available",
            "callback_id": "page_selection",
            "color": "#3AA3E3",
            "attachment_type": "default",
            "actions": buttons
        }]

        msg = {
            "response_type": "in_channel",
            "text": f"SQL Query (SQLite): ```{sql_query}```\nResult (rows {start_idx+1}-{min(end_idx, total_rows)} of {total_rows}):\n{formatted}\nPage {page_number} of {total_pages}",
            "attachments": attachments
        }
        requests.post(response_url, json=msg)

    threading.Thread(target=work).start()
    return jsonify({"response_type": "in_channel", "text": "Processing your request..."}), 200

# =====================
# ROUTE: Slack interactivity
# =====================
@app.route('/slack/interactions', methods=['POST'])
def slack_interactions():
    try:
        # Log the raw form payload from Slack
        print(">>> /slack/interactions HIT")
        print("RAW FORM:", dict(request.form))

        payload_raw = request.form.get('payload') or "{}"
        print("RAW payload string:", payload_raw)

        payload = json.loads(payload_raw)
        print("PARSED payload:", json.dumps(payload, indent=2))

        response_url = payload.get('response_url')
        actions = payload.get('actions', [])
        channel_id = payload.get('channel', {}).get('id')
        user_id = payload.get('user', {}).get('id')

        print("channel_id:", channel_id, "user_id:", user_id)
        print("actions:", actions)

        if not actions:
            print("No actions in payload")
            return jsonify({"text": "No actions found."}), 200

        for action in actions:
            # For Block Kit you'll get action_id; for legacy attachments you'll get name
            action_id = action.get('action_id') or action.get('name')
            value = action.get('value', '')

            print("ACTION_ID:", action_id, "VALUE:", value)

            if action_id in ['select_x_axis', 'select_y_axis']:
                selected = action['selected_option']['value']
                axis_type = 'X' if action_id == 'select_x_axis' else 'Y'
                store_axis_selection(user_id, axis_type, selected)
                return jsonify({"text": f"{axis_type} axis: {selected}"}), 200

            elif action_id == 'generate_plot_button':
                qtext = action.get('value') or ''  # original NL query travels here now
                threading.Thread(
                    target=generate_and_send_plot,
                    args=(user_id, response_url, payload, qtext),
                    daemon=True
                ).start()
                return jsonify({"text": "Generating plot..."}), 200

            elif action_id == 'export_csv':
                threading.Thread(
                    target=export_csv_from_interaction,
                    args=(value, response_url, payload),
                    daemon=True
                ).start()
                return jsonify({"text": "Exporting CSV..."}), 200

            elif action_id == 'plot':
                threading.Thread(
                    target=prepare_plot_from_query,
                    args=(value, response_url, payload),
                    daemon=True
                ).start()
                return jsonify({"text": "Preparing plot options..."}), 200

            elif action_id in ['next', 'previous']:
                page_str, query_text = value.split("|", 1)
                page_number = int(page_str)
                threading.Thread(
                    target=process_query_for_page,
                    args=(query_text, response_url, page_number),
                ).start()
                return jsonify({"text": f"Loading page {page_number}..."}), 200

        # 🔴 If we get here, no branch matched – this is what caused your 500.
        print("Unrecognized action. actions:", actions)
        return jsonify({"text": "Action received but no handler matched."}), 200

    except Exception as e:
        print("Interaction error:", e)
        return jsonify({"text": f"Error: {str(e)}"}), 500


    except Exception as e:
        print("Interaction error:", e)
        return jsonify({"text": f"Error: {str(e)}"}), 500

def prepare_plot_from_query(query_text, response_url, payload):
    user_id = payload['user']['id']

    sql_from_msg = extract_sql_from_original_message(payload)
    if sql_from_msg:
        csv_content = generate_csv(sql_query=sql_from_msg)
    else:
        csv_content = generate_csv(query_text=query_text)

    if not csv_content:
        requests.post(response_url, json={
            "response_type": "ephemeral",
            "text": "CSV generation failed while preparing plot."
        })
        return

    csv_path = save_csv_to_storage(csv_content, query_text)
    store_axis_selection(user_id, 'CSV', csv_path)

    columns = get_columns_from_csv(csv_path)
    if not columns:
        requests.post(response_url, json={"response_type":"ephemeral","text":"No columns found in CSV for plotting."})
        return

    blocks = [
        {"type":"section","text":{"type":"mrkdwn","text":"Select axes to generate a plot (CSV stored privately)."}},
        {"type":"actions","elements":[
            {"type":"static_select","placeholder":{"type":"plain_text","text":"Select X-axis"},
             "options":[{"text":{"type":"plain_text","text":c},"value":c} for c in columns],
             "action_id":"select_x_axis"},
            {"type":"static_select","placeholder":{"type":"plain_text","text":"Select Y-axis"},
             "options":[{"text":{"type":"plain_text","text":c},"value":c} for c in columns],
             "action_id":"select_y_axis"},
            {"type":"button","text":{"type":"plain_text","text":"Generate Plot"},
             "value": query_text,  # <— pass the original NL query here too
             "action_id":"generate_plot_button"}
        ]}
    ]
    requests.post(response_url, json={"response_type":"ephemeral","blocks": blocks})



def process_query_for_page(query_text, response_url, page_number):
    rows_per_page = 12
    sql_query = get_sql_query(query_text, DATABASE_SCHEMA)
    if not sql_query:
        requests.post(response_url, json={"response_type": "in_channel", "text": "I couldn't generate a safe SELECT."})
        return

    result = execute_sql_query(sql_query)
    if isinstance(result, dict) and "error" in result:
        requests.post(response_url, json={"response_type": "in_channel", "text": f"SQL error: {result['error']}"})
        return

    result_json = json.loads(result)
    total_rows = len(result_json)
    total_pages = (total_rows + rows_per_page - 1) // rows_per_page
    start = (page_number - 1) * rows_per_page
    end = start + rows_per_page
    slice_ = result_json[start:end]
    formatted = json_to_table(json.dumps(slice_))

    buttons = [
        make_button("export_csv", "Export as CSV", query_text),
        make_button("plot", "Bar Plot", query_text),
    ]
    if total_pages > 1:
        prev_page = max(1, page_number - 1)
        next_page = min(total_pages, page_number + 1)
        buttons.append(make_button("previous", f"Previous ({prev_page})", f"{prev_page}|{query_text}"))
        buttons.append(make_button("next", f"Next ({next_page})", f"{next_page}|{query_text}"))

    attachments = [{
        "text": "Navigate or export:",
        "fallback": "Actions not available",
        "callback_id": "page_selection",
        "color": "#3AA3E3",
        "attachment_type": "default",
        "actions": buttons
    }]

    msg = {
        "response_type": "in_channel",
        "text": f"SQL Query (SQLite): ```{sql_query}```\n{formatted}\nPage {page_number} of {total_pages}",
        "attachments": attachments
    }
    requests.post(response_url, json=msg)

# =================
# CSV + PLOTTING
# =================
def get_columns_from_csv(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)
        return list(df.columns)
    except Exception as e:
        print(f"CSV read error {csv_filepath}: {e}")
        return []

def save_csv_to_storage(csv_content, query_text):
    name = hashlib.md5(query_text.encode()).hexdigest() + ".csv"
    path = os.path.join(EXPORTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    return path

def generate_csv(query_text=None, sql_query=None):
    """
    Generate a CSV from either:
      - an existing SQL query (preferred), or
      - an NL question via the LLM (fallback).
    """
    if sql_query is None:
        if not query_text:
            return None
        sql_query = get_sql_query(query_text, DATABASE_SCHEMA)

    if not sql_query:
        return None

    result = execute_sql_query(sql_query)
    if isinstance(result, dict) and "error" in result:
        print("generate_csv SQL error:", result["error"])
        return None

    try:
        df = pd.DataFrame(json.loads(result))
    except Exception as e:
        print("generate_csv JSON/DF error:", e)
        return None

    if df.empty:
        # You might still want a CSV with just headers, but this keeps your old behaviour.
        return df.to_csv(index=False)
    return df.to_csv(index=False)


def export_csv_from_interaction(query_text, response_url, payload):
    user_id = payload['user']['id']
    channel_id = payload.get('channel', {}).get('id')

    # Prefer the exact SQL used in the original message
    sql_from_msg = extract_sql_from_original_message(payload)
    if sql_from_msg:
        csv_content = generate_csv(sql_query=sql_from_msg)
    else:
        # Fallback: regenerate from the NL question if needed
        csv_content = generate_csv(query_text=query_text)

    if not csv_content:
        requests.post(response_url, json={
            "response_type": "ephemeral",
            "text": "CSV generation failed (could not produce SQL or query returned an error)."
        })
        return

    csv_path = save_csv_to_storage(csv_content, query_text)

    # ⬇️ OPTIONAL: you can skip storing CSV here since plotting uses the 'plot' flow
    # store_axis_selection(user_id, 'CSV', csv_path)

    # Upload CSV to Slack
    with open(csv_path, "rb") as f:
        ok, resp = slack_upload_file(
            channel_id,
            f.read(),
            filename=os.path.basename(csv_path),
            title="Query Export",
            initial_comment="Here is your CSV file."
        )

    if not ok:
        requests.post(response_url, json={
            "response_type": "ephemeral",
            "text": f"Failed to upload CSV to Slack: `{resp}`"
        })
        return

    # ✅ Done: just confirm upload, no axis-picker UI
    requests.post(response_url, json={
        "response_type": "ephemeral",
        "text": "CSV uploaded ✅"
    })





def generate_and_send_plot(user_id, response_url, payload, query_text=""):
    x_axis, y_axis, csv_filepath = fetch_axis_selection(user_id)
    channel_id = payload.get('channel', {}).get('id')

    # If CSV vanished or server restarted, rebuild it from the query that came with the button.
    if (not csv_filepath or not os.path.exists(csv_filepath)) and query_text:
        csv_content = generate_csv(query_text)
        if csv_content:
            csv_filepath = save_csv_to_storage(csv_content, query_text)
            store_axis_selection(user_id, 'CSV', csv_filepath)

    if not csv_filepath or not os.path.exists(csv_filepath):
        requests.post(response_url, json={"text": "CSV not found for plotting. Click *Bar Plot* again."})
        return

    if not x_axis or not y_axis:
        requests.post(response_url, json={"text": "Please select both X and Y axes first."})
        return

    df = pd.read_csv(csv_filepath)
    if df.empty or x_axis not in df.columns or y_axis not in df.columns:
        requests.post(response_url, json={"text": "Invalid axes or no data to plot."})
        return

    # (Optional) aggregate for repeated X to make bar clearer
    if df.duplicated(subset=[x_axis]).any():
        df = df.groupby(x_axis, dropna=False)[y_axis].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(df[x_axis].astype(str), pd.to_numeric(df[y_axis], errors='coerce'))
    plt.xticks(rotation=45, ha='right'); plt.xlabel(x_axis); plt.ylabel(y_axis)
    plt.title(f"{y_axis} by {x_axis}"); plt.tight_layout()

    plot_path = os.path.join(EXPORTS_DIR, f"plot_{user_id}_{int(time.time())}.png")
    plt.savefig(plot_path); plt.close()

    with open(plot_path, "rb") as f:
        ok, resp = slack_upload_file(
            channel_id, f.read(),
            filename=os.path.basename(plot_path),
            title="Plot",
            initial_comment=f"Here is your plot for *{y_axis}* by *{x_axis}*."
        )
    if not ok:
        requests.post(response_url, json={"text": f"Failed to upload plot to Slack: {resp}"})
        return

    requests.post(response_url, json={"text": "Plot uploaded ✅"})


# =================
# /slack/help route
# =================
@app.route('/slack/help', methods=['POST'])
def help_command():
    data = request.form
    response_url = data.get('response_url')
    msg = {
        "response_type": "in_channel",
        "text": "How to use the SQLite + Ollama Assistant",
        "attachments": [{
            "text": "Commands:",
            "color": "#36a64f",
            "fields": [
                {"title": "Custom Query", "value": "`/dd [your question]` (e.g. 'top 10 customers by spend')", "short": False},
                {"title": "Paging", "value": "`/dd [your question] <page_number>` to jump pages", "short": False},
                {"title": "Export CSV", "value": "Use *Export as CSV* button (uploads file to Slack)", "short": False},
                {"title": "Plot", "value": "Use *Plot Data* → choose axes → *Generate Plot* (uploads image)", "short": False},
            ]
        }]
    }
    requests.post(response_url, json=msg)
    return '', 204

# ===============
# (Optional) RAG
# ===============
# If you still need /slack/onboarding or /slack/abbreviations, keep your existing handlers.
# They are left out here to focus on the DB + LLM refactor.

# =========
# MAIN
# =========
if __name__ == '__main__':
    # Basic sanity prints
    print("Using SQLite at:", SQLITE_PATH)
    print("Using Ollama model:", OLLAMA_MODEL, "at", OLLAMA_BASE_URL)
    print("Schema summary:\n", DATABASE_SCHEMA[:600], "...\n")
    app.run(host="0.0.0.0", port=5000, debug=True)

