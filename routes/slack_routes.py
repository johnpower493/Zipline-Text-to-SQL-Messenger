"""
Flask routes for Slack interactions.
"""
import json
import os
import time
import threading
from flask import Blueprint, request, jsonify, send_from_directory
from typing import Dict, Any

from config.settings import config
from services.database import DatabaseService
from services.llm import LLMService
from services.slack import SlackService
from services.data_export import DataExportService, session_manager
from services.true_agentic_analyst import true_agentic_analyst
from utils.formatting import (
    format_data_as_table, 
    paginate_query_results, 
    calculate_pagination_info,
    extract_sql_from_slack_message,
    parse_page_command
)
from models.slack import (
    QueryRequest, 
    SlackAction, 
    create_slack_button, 
    create_slack_attachment_with_buttons
)

# Create Blueprint
slack_bp = Blueprint('slack', __name__, url_prefix='/slack')

# Get database schema once at startup
DATABASE_SCHEMA = DatabaseService.get_database_schema()


@slack_bp.route('/sqlquery', methods=['POST'])
def handle_sql_query():
    """Handle the /dd slash command for SQL queries."""
    try:
        # Parse request data
        form_data = request.form
        command_text = form_data.get('text', '').strip()
        
        # Extract query and page number
        query_text, page_number = parse_page_command(command_text)
        
        # Create query request
        query_request = QueryRequest(
            question=query_text,
            page_number=page_number,
            user_id=form_data.get('user_id', ''),
            channel_id=form_data.get('channel_id', ''),
            response_url=form_data.get('response_url', '')
        )
        
        # Process in background thread
        threading.Thread(
            target=_process_sql_query,
            args=(query_request,),
            daemon=True
        ).start()
        
        return jsonify({
            "response_type": "in_channel",
            "text": "Processing your request..."
        }), 200
        
    except Exception as e:
        print(f"Error in handle_sql_query: {e}")
        return jsonify({"text": f"Error processing request: {str(e)}"}), 500


@slack_bp.route('/interactions', methods=['POST'])
def handle_slack_interactions():
    """Handle Slack button interactions and form submissions."""
    try:
        # Parse the payload
        payload_raw = request.form.get('payload', '{}')
        payload = json.loads(payload_raw)
        
        print(f"Slack interaction: {json.dumps(payload, indent=2)}")
        
        # Extract common data
        response_url = payload.get('response_url', '')
        actions = payload.get('actions', [])
        user_id = payload.get('user', {}).get('id', '')
        channel_id = payload.get('channel', {}).get('id', '')
        
        if not actions:
            return jsonify({"text": "No actions found."}), 200
        
        # Process each action
        for action_data in actions:
            action = SlackAction.from_slack_payload(action_data)
            
            if action.action_id in ['select_x_axis', 'select_y_axis']:
                # Handle axis selection
                axis_type = 'X' if action.action_id == 'select_x_axis' else 'Y'
                session_manager.store_user_selection(user_id, axis_type, action.value)
                
                return jsonify({"text": f"{axis_type} axis: {action.value}"}), 200
                
            elif action.action_id == 'generate_plot_button':
                # Handle plot generation
                threading.Thread(
                    target=_generate_plot,
                    args=(user_id, response_url, channel_id, action.value),
                    daemon=True
                ).start()
                return jsonify({"text": "Generating plot..."}), 200
                
            elif action.action_id == 'export_csv':
                # Handle CSV export
                threading.Thread(
                    target=_export_csv,
                    args=(action.value, response_url, payload),
                    daemon=True
                ).start()
                return jsonify({"text": "Exporting CSV..."}), 200
                
            elif action.action_id in ['plot_bar', 'plot_line', 'plot_pie']:
                # Handle plot preparation with plot type
                if action.action_id == 'plot_bar':
                    plot_type = 'bar'
                elif action.action_id == 'plot_line':
                    plot_type = 'line'
                else:  # plot_pie
                    plot_type = 'pie'
                
                threading.Thread(
                    target=_prepare_plot,
                    args=(action.value, response_url, payload, plot_type),
                    daemon=True
                ).start()
                return jsonify({"text": f"Preparing {plot_type} plot options..."}), 200
                
            elif action.action_id == 'insights':
                # Handle insights generation
                threading.Thread(
                    target=_generate_insights,
                    args=(action.value, response_url, payload),
                    daemon=True
                ).start()
                return jsonify({"text": "🔍 Generating insights..."}), 200
            
            elif action.action_id == "page_info":
                # This is just a visual indicator, ignore clicks
                return jsonify({"response_type": "ephemeral", "text": "This shows the current page number."}), 200
                
            elif action.action_id in ['next', 'previous']:
                # Handle pagination
                try:
                    page_str, query_text = action.value.split("|", 1)
                    page_number = int(page_str)
                    
                    query_request = QueryRequest(
                        question=query_text,
                        page_number=page_number,
                        response_url=response_url
                    )
                    
                    threading.Thread(
                        target=_process_sql_query,
                        args=(query_request,),
                        daemon=True
                    ).start()
                    
                    return jsonify({"text": f"Loading page {page_number}..."}), 200
                    
                except (ValueError, IndexError) as e:
                    return jsonify({"text": f"Invalid page navigation: {str(e)}"}), 400
        
        return jsonify({"text": "Action processed"}), 200
        
    except Exception as e:
        print(f"Error in handle_slack_interactions: {e}")
        return jsonify({"text": f"Error processing interaction: {str(e)}"}), 500


@slack_bp.route('/askdb', methods=['POST'])
def askdb_command():
    """Handle the /askdb agentic assistant command."""
    try:
        # Validate request
        if not request.form.get("token"):
            return jsonify({"error": "Invalid request"}), 400
        
        question = request.form.get("text", "").strip()
        user_id = request.form.get("user_id", "unknown")
        response_url = request.form.get("response_url")
        
        if not question:
            return jsonify({
                "response_type": "ephemeral",
                "text": "Please provide a business question to investigate.\n\n*Examples:*\n• `why did sales drop last month?`\n• `who are my top customers?`\n• `which products are most profitable?`\n• `what trends do I see in customer behavior?`"
            })
        
        # Create request object for the investigation
        investigation_request = QueryRequest(
            question=question,
            user_id=user_id,
            channel_id=request.form.get("channel_id", ""),
            response_url=response_url
        )
        
        # Process investigation in background thread
        threading.Thread(
            target=_process_askdb_investigation,
            args=(investigation_request,),
            daemon=True
        ).start()
        
        return jsonify({
            "response_type": "in_channel",
            "text": f"🤖 *Investigating:* {question}\n\n_Analyzing your data across multiple dimensions..._"
        })
        
    except Exception as e:
        print(f"Error in askdb handler: {e}")
        return jsonify({
            "response_type": "ephemeral", 
            "text": "Sorry, I encountered an error processing your question."
        }), 500


@slack_bp.route('/help', methods=['POST'])
def help_command():
    """Handle the /help slash command."""
    try:
        response_url = request.form.get('response_url')
        
        help_message = {
            "response_type": "in_channel",
            "text": "🤖 How to use CircularQuery - Your AI Data Assistant",
            "attachments": [{
                "text": "Commands:",
                "color": "#36a64f",
                "fields": [
                    {
                        "title": "/dd Command",
                        "value": "`/dd [your question]` → Get SQL + table results + actions (e.g. 'top 10 customers by spend')",
                        "short": False
                    },
                    {
                        "title": "/askdb Command (NEW!)",
                        "value": "`/askdb [business question]` → AI investigates using multiple queries + comprehensive analysis (e.g. 'why did sales drop last month?')",
                        "short": False
                    },
                    {
                        "title": "Paging",
                        "value": "`/dd [your question] <page_number>` to jump pages",
                        "short": False
                    },
                    {
                        "title": "Export CSV",
                        "value": "Use *Export as CSV* button (uploads file to Slack)",
                        "short": False
                    },
                    {
                        "title": "Plots",
                        "value": "Use *📊 Bar Plot*, *📈 Line Plot*, or *🥧 Pie Chart* → choose data columns → *Generate Plot* (uploads image)",
                        "short": False
                    },
                    {
                        "title": "Insights",
                        "value": "Use *🔍 Insights* button to get AI-powered data analysis and recommendations",
                        "short": False
                    },
                ]
            }]
        }
        
        if response_url:
            import requests
            requests.post(response_url, json=help_message)
        
        return '', 204
        
    except Exception as e:
        print(f"Error in help_command: {e}")
        return jsonify({"text": "Error displaying help"}), 500


@slack_bp.route('/exports/<filename>')
def download_file(filename):
    """Serve exported files (for local development)."""
    return send_from_directory(config.EXPORTS_DIR, filename)


# Background processing functions

def _process_sql_query(query_request: QueryRequest):
    """Process SQL query in background thread."""
    import requests
    
    try:
        print(f"[DEBUG] Processing query: '{query_request.question}' for user {query_request.user_id}")
        print(f"[DEBUG] Response URL: {query_request.response_url}")
        print(f"[DEBUG] LLM Backend: {config.LLM_BACKEND}")
        
        start_time = time.time()
        # Generate SQL from natural language
        print(f"[DEBUG] Generating SQL with {config.LLM_BACKEND}...")
        llm_start = time.time()
        sql_query = LLMService.get_sql_query(query_request.question, DATABASE_SCHEMA)
        llm_time = time.time() - llm_start
        print(f"[DEBUG] LLM took {llm_time:.2f}s, generated: {sql_query}")
        
        if not sql_query:
            error_response = {
                "response_type": "in_channel",
                "text": "I couldn't generate a safe SELECT statement for SQLite. Try rephrasing your question."
            }
            print(f"[DEBUG] Posting error response to: {query_request.response_url}")
            response = requests.post(query_request.response_url, json=error_response, timeout=10)
            print(f"[DEBUG] Error response status: {response.status_code}")
            return
        
        # Execute query
        result = DatabaseService.execute_query(sql_query)
        
        if "error" in result:
            requests.post(query_request.response_url, json={
                "response_type": "in_channel",
                "text": f"SQL error: {result['error']}"
            })
            return
        
        data = result["data"]
        
        if not data:
            requests.post(query_request.response_url, json={
                "response_type": "in_channel",
                "text": f"SQL Query: ```{sql_query}```\nNo data found."
            })
            return
        
        # Calculate pagination
        pagination = calculate_pagination_info(len(data), query_request.page_number, config.ROWS_PER_PAGE)
        
        # Get page slice
        page_data = paginate_query_results(data, query_request.page_number, config.ROWS_PER_PAGE)
        formatted_table = format_data_as_table(page_data)
        
        # Create action buttons in 2 clean rows
        # Row 1: Main actions (max 5 buttons)
        row1_buttons = [
            create_slack_button("export_csv", "📎 Export CSV", query_request.question),
            create_slack_button("plot_bar", "📊 Bar Chart", query_request.question),
            create_slack_button("plot_line", "📈 Line Chart", query_request.question),
            create_slack_button("plot_pie", "🥧 Pie Chart", query_request.question),
            create_slack_button("insights", "🔍 Insights", query_request.question, "default"),
        ]
        
        # Row 2: Pagination (only if needed)
        row2_buttons = []
        if pagination["total_pages"] > 1:
            if pagination["has_previous"]:
                row2_buttons.append(create_slack_button(
                    "previous", 
                    f"⬅️ Page {pagination['previous_page']}", 
                    f"{pagination['previous_page']}|{query_request.question}"
                ))
            
            # Add page indicator (non-clickable visual element)
            row2_buttons.append({
                "type": "button",
                "text": {"type": "plain_text", "text": f"📄 {pagination['current_page']} of {pagination['total_pages']}"},
                "action_id": "page_info",
                "style": "primary"
            })
            
            if pagination["has_next"]:
                row2_buttons.append(create_slack_button(
                    "next", 
                    f"Page {pagination['next_page']} ➡️", 
                    f"{pagination['next_page']}|{query_request.question}"
                ))
        
        # Create clean attachments
        attachments = []
        attachments.append(create_slack_attachment_with_buttons("", row1_buttons))  # No label needed
        
        if row2_buttons:
            attachments.append(create_slack_attachment_with_buttons("", row2_buttons))
        
        message = {
            "response_type": "in_channel",
            "text": (
                f"SQL Query (SQLite): ```{sql_query}```\n"
                f"Result (rows {pagination['start_row']}-{pagination['end_row']} of {pagination['total_rows']}):\n"
                f"{formatted_table}\n"
                f"Page {pagination['current_page']} of {pagination['total_pages']}"
            ),
            "attachments": attachments
        }
        
        print(f"[DEBUG] Posting success response to: {query_request.response_url}")
        print(f"[DEBUG] Total processing time: {time.time() - start_time:.2f}s")
        response = requests.post(query_request.response_url, json=message, timeout=10)
        print(f"[DEBUG] Success response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[DEBUG] Response error: {response.text}")
        
    except Exception as e:
        print(f"Error processing SQL query: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            error_response = {
                "response_type": "ephemeral",
                "text": f"Error processing query: {str(e)}"
            }
            response = requests.post(query_request.response_url, json=error_response, timeout=10)
            print(f"[DEBUG] Exception response status: {response.status_code}")
        except Exception as post_error:
            print(f"Failed to post error response: {post_error}")


def _export_csv(query_text: str, response_url: str, payload: Dict[str, Any]):
    """Export CSV in background thread."""
    import requests
    
    try:
        channel_id = payload.get('channel', {}).get('id', '')
        
        # Try to get SQL from original message, otherwise regenerate
        sql_query = extract_sql_from_slack_message(payload)
        if not sql_query:
            sql_query = LLMService.get_sql_query(query_text, DATABASE_SCHEMA)
        
        if not sql_query:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": "Could not generate SQL for CSV export."
            })
            return
        
        # Execute query
        result = DatabaseService.execute_query(sql_query)
        if "error" in result:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": f"SQL error during CSV export: {result['error']}"
            })
            return
        
        # Generate CSV
        csv_content = DataExportService.generate_csv_from_data(result["data"])
        if not csv_content:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": "Failed to generate CSV content."
            })
            return
        
        # Save and upload CSV
        csv_filepath = DataExportService.save_csv_to_storage(csv_content, query_text)
        
        with open(csv_filepath, "rb") as f:
            success, response = SlackService.upload_file(
                channel_id,
                f.read(),
                filename=f"query_export_{int(time.time())}.csv",
                title="Query Export",
                initial_comment="Here is your CSV export."
            )
        
        if success:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": "CSV uploaded successfully ✅"
            })
        else:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": f"Failed to upload CSV: {response}"
            })
            
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        requests.post(response_url, json={
            "response_type": "ephemeral",
            "text": f"Error exporting CSV: {str(e)}"
        })


def _process_askdb_investigation(investigation_request: QueryRequest):
    """Process business question investigation in background thread."""
    import requests
    
    try:
        print(f"[DEBUG] Processing askdb investigation: '{investigation_request.question}' for user {investigation_request.user_id}")
        print(f"[DEBUG] Response URL: {investigation_request.response_url}")
        print(f"[DEBUG] LLM Backend: {config.LLM_BACKEND}")
        
        start_time = time.time()
        
        # Get database schema
        schema = DatabaseService.get_database_schema()
        print(f"[DEBUG] Schema length: {len(schema)} characters")
        print(f"[DEBUG] Schema preview: {schema[:500]}...")
        
        # Test basic database connectivity
        test_result = DatabaseService.execute_query("SELECT COUNT(*) AS test_count FROM drivers")
        print(f"[DEBUG] Test query result: {test_result}")
        
        # Use TRUE agentic analyst to investigate the question
        print(f"[DEBUG] Starting TRUE agentic investigation...")
        analysis_result = true_agentic_analyst.investigate(investigation_request.question, schema)
        
        investigation_time = time.time() - start_time
        print(f"[DEBUG] Investigation completed in {investigation_time:.2f}s")
        
        # Convert Markdown to Slack formatting
        slack_formatted_result = _convert_markdown_to_slack(analysis_result)
        
        # Create response message
        message = {
            "response_type": "in_channel",
            "text": slack_formatted_result
        }
        
        print(f"[DEBUG] Analysis result preview: {analysis_result[:300]}...")
        print(f"[DEBUG] Analysis result length: {len(analysis_result)} characters")
        print(f"[DEBUG] Posting investigation result to: {investigation_request.response_url}")
        response = requests.post(investigation_request.response_url, json=message, timeout=10)
        print(f"[DEBUG] Investigation response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[DEBUG] Response error: {response.text}")
        
    except Exception as e:
        print(f"Error processing askdb investigation: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            error_response = {
                "response_type": "in_channel",
                "text": f"❌ **Investigation Failed**\n\nI encountered an error while investigating: *{investigation_request.question}*\n\nError: {str(e)}\n\nPlease try rephrasing your question or contact support."
            }
            response = requests.post(investigation_request.response_url, json=error_response, timeout=10)
            print(f"[DEBUG] Error response status: {response.status_code}")
        except Exception as post_error:
            print(f"Failed to post investigation error response: {post_error}")


def _convert_markdown_to_slack(markdown_text: str) -> str:
    """Convert Markdown formatting to Slack formatting."""
    import re
    
    # Convert markdown to Slack formatting
    slack_text = markdown_text
    
    # Convert **bold** to *bold*
    slack_text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', slack_text)
    
    # Convert bullet points • to Slack format
    slack_text = re.sub(r'^•\s*', '• ', slack_text, flags=re.MULTILINE)
    
    # Convert section headers with emojis
    slack_text = re.sub(r'^🤖 \*\*(.*?)\*\*', r'🤖 *\1*', slack_text, flags=re.MULTILINE)
    slack_text = re.sub(r'^📊 \*\*(.*?)\*\*:', r'📊 *\1*:', slack_text, flags=re.MULTILINE)
    slack_text = re.sub(r'^🔍 \*\*(.*?)\*\*:', r'🔍 *\1*:', slack_text, flags=re.MULTILINE)
    slack_text = re.sub(r'^💡 \*\*(.*?)\*\*:', r'💡 *\1*:', slack_text, flags=re.MULTILINE)
    slack_text = re.sub(r'^📈 \*\*(.*?)\*\*:', r'📈 *\1*:', slack_text, flags=re.MULTILINE)
    
    # Clean up any remaining ** formatting
    slack_text = slack_text.replace('**', '*')
    
    return slack_text


def _prepare_plot(query_text: str, response_url: str, payload: Dict[str, Any], plot_type: str = 'bar'):
    """Prepare plot options in background thread."""
    import requests
    
    try:
        user_id = payload.get('user', {}).get('id', '')
        
        # Generate CSV for plotting
        sql_query = extract_sql_from_slack_message(payload)
        if not sql_query:
            sql_query = LLMService.get_sql_query(query_text, DATABASE_SCHEMA)
        
        if not sql_query:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": "Could not generate SQL for plotting."
            })
            return
        
        result = DatabaseService.execute_query(sql_query)
        if "error" in result:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": f"SQL error: {result['error']}"
            })
            return
        
        csv_content = DataExportService.generate_csv_from_data(result["data"])
        if not csv_content:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": "No data available for plotting."
            })
            return
        
        csv_filepath = DataExportService.save_csv_to_storage(csv_content, query_text)
        session_manager.store_user_selection(user_id, "CSV", csv_filepath)
        session_manager.store_user_selection(user_id, "PLOT_TYPE", plot_type)
        
        columns = DataExportService.get_csv_columns(csv_filepath)
        if not columns:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": "No columns found for plotting."
            })
            return
        
        # Create Block Kit UI for axis selection - different for pie charts
        if plot_type == 'pie':
            # Pie chart needs Category and Value columns
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Select data for pie chart:"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "static_select",
                            "placeholder": {"type": "plain_text", "text": "Select Category (labels)"},
                            "options": [
                                {"text": {"type": "plain_text", "text": col}, "value": col}
                                for col in columns
                            ],
                            "action_id": "select_x_axis"  # Reuse X-axis for category
                        },
                        {
                            "type": "static_select",
                            "placeholder": {"type": "plain_text", "text": "Select Values (sizes)"},
                            "options": [
                                {"text": {"type": "plain_text", "text": col}, "value": col}
                                for col in columns
                            ],
                            "action_id": "select_y_axis"  # Reuse Y-axis for values
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Generate Pie Chart 🥧"},
                            "value": query_text,
                            "action_id": "generate_plot_button"
                        }
                    ]
                }
            ]
        else:
            # Regular axis selection for bar/line plots
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Select axes to generate a plot:"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "static_select",
                            "placeholder": {"type": "plain_text", "text": "Select X-axis"},
                            "options": [
                                {"text": {"type": "plain_text", "text": col}, "value": col}
                                for col in columns
                            ],
                            "action_id": "select_x_axis"
                        },
                        {
                            "type": "static_select",
                            "placeholder": {"type": "plain_text", "text": "Select Y-axis"},
                            "options": [
                                {"text": {"type": "plain_text", "text": col}, "value": col}
                                for col in columns
                            ],
                            "action_id": "select_y_axis"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": f"Generate {plot_type.title()} Plot"},
                            "value": query_text,
                            "action_id": "generate_plot_button"
                        }
                    ]
                }
            ]
        
        requests.post(response_url, json={
            "response_type": "ephemeral",
            "blocks": blocks
        })
        
    except Exception as e:
        print(f"Error preparing plot: {e}")
        requests.post(response_url, json={
            "response_type": "ephemeral",
            "text": f"Error preparing plot: {str(e)}"
        })


def _generate_plot(user_id: str, response_url: str, channel_id: str, query_text: str):
    """Generate and upload plot in background thread."""
    import requests
    import time
    
    try:
        # Get user selections including plot type
        x_axis, y_axis, csv_filepath = session_manager.get_user_selections(user_id)
        plot_type = session_manager.get_user_selection(user_id, "PLOT_TYPE") or "bar"
        
        # Rebuild CSV if needed
        if not csv_filepath or not os.path.exists(csv_filepath):
            if query_text:
                sql_query = LLMService.get_sql_query(query_text, DATABASE_SCHEMA)
                if sql_query:
                    result = DatabaseService.execute_query(sql_query)
                    if "data" in result:
                        csv_content = DataExportService.generate_csv_from_data(result["data"])
                        csv_filepath = DataExportService.save_csv_to_storage(csv_content, query_text)
                        session_manager.store_user_selection(user_id, "CSV", csv_filepath)
        
        if not csv_filepath or not os.path.exists(csv_filepath):
            requests.post(response_url, json={
                "text": "CSV not found for plotting. Please try the Plot Data button again."
            })
            return
        
        if not x_axis or not y_axis:
            requests.post(response_url, json={
                "text": "Please select both X and Y axes first."
            })
            return
        
        # Create plot based on type
        if plot_type == "line":
            plot_filepath = DataExportService.create_line_plot(csv_filepath, x_axis, y_axis, user_id)
        elif plot_type == "pie":
            plot_filepath = DataExportService.create_pie_chart(csv_filepath, x_axis, y_axis, user_id)
        else:  # bar
            plot_filepath = DataExportService.create_bar_plot(csv_filepath, x_axis, y_axis, user_id)
        
        if not plot_filepath:
            requests.post(response_url, json={
                "text": "Failed to create plot. Please check your axis selections."
            })
            return
        
        # Upload plot to Slack
        with open(plot_filepath, "rb") as f:
            success, response = SlackService.upload_file(
                channel_id,
                f.read(),
                filename=f"plot_{int(time.time())}.png",
                title="Data Plot",
                initial_comment=f"Plot of *{y_axis}* by *{x_axis}*"
            )
        
        if success:
            requests.post(response_url, json={"text": "Plot uploaded successfully ✅"})
        else:
            requests.post(response_url, json={"text": f"Failed to upload plot: {response}"})
            
        # Clean up
        try:
            os.remove(plot_filepath)
        except:
            pass
            
    except Exception as e:
        print(f"Error generating plot: {e}")
        requests.post(response_url, json={"text": f"Error generating plot: {str(e)}"})


def _generate_insights(query_text: str, response_url: str, payload: Dict[str, Any]):
    """Generate data insights in background thread."""
    import requests
    
    try:
        # Try to get SQL from original message, otherwise regenerate
        sql_query = extract_sql_from_slack_message(payload)
        if not sql_query:
            sql_query = LLMService.get_sql_query(query_text, DATABASE_SCHEMA)
        
        if not sql_query:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": "Could not generate SQL for insights analysis."
            })
            return
        
        # Execute query to get fresh data
        result = DatabaseService.execute_query(sql_query)
        if "error" in result:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": f"SQL error during insights generation: {result['error']}"
            })
            return
        
        data = result["data"]
        if not data:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": "No data available for insights analysis."
            })
            return
        
        # Generate insights using LLM
        insights = LLMService.generate_insights(query_text, sql_query, data, DATABASE_SCHEMA)
        
        if not insights:
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": "Could not generate insights. Please try again or rephrase your question."
            })
            return
        
        # Format the insights response with proper Slack markdown
        formatted_insights = _format_insights_for_slack(insights)
        
        insights_message = {
            "response_type": "in_channel",
            "text": f"🔍 *Data Insights for:* _{query_text}_",
            "attachments": [{
                "color": "#36C5F0",
                "text": formatted_insights,
                "mrkdwn_in": ["text"],
                "footer": f"Based on {len(data)} rows of data",
                "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png"
            }]
        }
        
        requests.post(response_url, json=insights_message)
        
    except Exception as e:
        print(f"Error generating insights: {e}")
        requests.post(response_url, json={
            "response_type": "ephemeral",
            "text": f"Error generating insights: {str(e)}"
        })


def _format_insights_for_slack(insights: str) -> str:
    """
    Convert standard Markdown to Slack's mrkdwn format.
    
    Args:
        insights: Raw insights text with potential Markdown formatting
        
    Returns:
        Formatted text using Slack's mrkdwn syntax
    """
    if not insights:
        return insights
    
    # Convert common Markdown patterns to Slack mrkdwn
    formatted = insights
    
    # Convert **bold** to *bold* (Slack uses single asterisks for bold)
    import re
    formatted = re.sub(r'\*\*(.*?)\*\*', r'*\1*', formatted)
    
    # Convert __bold__ to *bold* as well
    formatted = re.sub(r'__(.*?)__', r'*\1*', formatted)
    
    # Convert *italic* to _italic_ (Slack uses underscores for italic)
    # But be careful not to convert the bold asterisks we just created
    # Use negative lookbehind/lookahead to avoid converting bold markers
    formatted = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'_\1_', formatted)
    
    # Convert headers (# Header) to bold with line breaks
    formatted = re.sub(r'^#+\s*(.+?)$', r'*\1*', formatted, flags=re.MULTILINE)
    
    # Convert bullet points to Slack format
    formatted = re.sub(r'^[\-\*\+]\s+', '• ', formatted, flags=re.MULTILINE)
    
    # Convert numbered lists to Slack format (keep the numbers)
    formatted = re.sub(r'^\d+\.\s+', lambda m: f"{m.group(0)}", formatted, flags=re.MULTILINE)
    
    # Clean up excessive line breaks (max 2 consecutive)
    formatted = re.sub(r'\n\s*\n\s*\n+', '\n\n', formatted)
    
    # Ensure proper spacing around sections
    formatted = re.sub(r'\n(\*[^*]+\*)\n', r'\n\n\1\n', formatted)
    
    return formatted.strip()