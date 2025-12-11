"""
LLM service for generating SQL queries using Ollama or Groq.
"""
import requests
import re
from typing import Optional
from config.settings import config
from core.guardrails import sanitize_sql
from services.database import DatabaseService

# Optional import for Groq support
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMService:
    """Handles LLM operations for SQL generation."""
    
    @staticmethod
    def _extract_constructor_keyword(text: str) -> Optional[str]:
        if not text:
            return None
        t = text.lower()
        teams = [
            "red bull", "mercedes", "ferrari", "mclaren", "aston martin", "alpine",
            "williams", "haas", "alphatauri", "rb", "sauber", "andretti"
        ]
        for team in teams:
            if team in t:
                return team
        return None

    @staticmethod
    def _sql_has_constructor_filter(sql: str, team_kw: str) -> bool:
        s = (sql or "").lower()
        if "constructors" not in s:
            return False
        if team_kw and team_kw not in s:
            # allow partial match on constructorRef-like tokens
            ref = team_kw.replace(" ", "_")
            if ref not in s:
                return False
        return True

    @staticmethod
    def _get_groq_client() -> Optional[OpenAI]:
        """Get configured Groq client."""
        if not OpenAI or not config.GROQ_API_KEY:
            return None
        return OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    
    @staticmethod
    def generate_prompt(question: str, schema: str) -> str:
        """
        Generate a robust prompt for SQL generation with clear constraints and hints.
        """
        guidance = (
            "You are a strict SQLite SQL generator. "
            "Return ONLY a valid SQLite SELECT statement. No explanations. No comments. No markdown fences. "
            "Use only tables and columns from the provided schema context. Prefer explicit JOIN ... ON ... using the foreign keys shown. "
            "Use aliases sparingly and include a LIMIT if appropriate."
        )
        join_hints = (
            "Guidelines: Use INNER JOIN or LEFT JOIN as needed; join keys are given by FKs[...] in the schema context. "
            "Honor entity filters in the question (e.g., constructor/team names, driver names, seasons/years). Join drivers/constructors/races/results appropriately to enforce filters. "
            "Avoid subqueries unless necessary. Avoid window functions unless asked."
        )
        domain_hints = LLMService._build_f1_hints_and_examples(schema, question)
        domain_block = f"\n\nDOMAIN HINTS:\n{domain_hints}\n" if domain_hints else ""
        return (
            f"{guidance}\n\n"
            f"SCHEMA CONTEXT:\n{schema}\n\n"
            f"JOIN HINTS:\n{join_hints}{domain_block}\n"
            f"Question: {question}\n"
            "SQL:"
        )

    @staticmethod
    def _build_f1_hints_and_examples(schema: str, question: str) -> str:
        """Return F1-specific join hints and few-shot examples when F1 tables are present."""
        s = schema.lower()
        q = (question or "").lower()
        required = all(tok in s for tok in ["drivers(", "results(", "races(", "constructors("])
        if not required:
            return ""
        # basic detector for relevance
        if not any(k in q for k in ["driver", "constructor", "race", "points", "season", "year", "grand prix", "red bull", "mercedes", "ferrari"]):
            # still show hints if schema is F1, but keep examples minimal
            show_examples = False
        else:
            show_examples = True
        lines = []
        lines.append("Common F1 joins:")
        lines.append("- results.driverId = drivers.driverId")
        lines.append("- results.raceId = races.raceId")
        lines.append("- results.constructorId = constructors.constructorId")
        lines.append("- driver_standings.driverId = drivers.driverId (if available)")
        lines.append("- driver_standings.raceId = races.raceId (if available)")
        lines.append("- constructor_standings.constructorId = constructors.constructorId (if available)")
        lines.append("- constructor_standings.raceId = races.raceId (if available)")
        lines.append("")
        lines.append("Filters:")
        lines.append("- Season/year: filter on races.year = <year>")
        lines.append("- Team/constructor: filter via constructors.name or constructors.constructorRef after joining through results/standings")
        lines.append("- Driver name: use drivers.forename/surname/code")
        if show_examples:
            lines.append("")
            lines.append("Examples (adapt as needed):")
            # Example 1: Red Bull driver points 2023
            lines.append("-- Red Bull driver total points in 2023")
            lines.append("SELECT d.driverId, d.forename || ' ' || d.surname AS driver_name, SUM(ds.points) AS total_points")
            lines.append("FROM driver_standings ds")
            lines.append("JOIN races r ON r.raceId = ds.raceId")
            lines.append("JOIN drivers d ON d.driverId = ds.driverId")
            lines.append("JOIN results res ON res.driverId = d.driverId AND res.raceId = r.raceId")
            lines.append("JOIN constructors c ON c.constructorId = res.constructorId")
            lines.append("WHERE r.year = 2023 AND c.name LIKE '%Red Bull%' ")
            lines.append("GROUP BY d.driverId, driver_name")
            lines.append("ORDER BY total_points DESC")
            lines.append("LIMIT 10")
            # Example 2: Constructor points by race in a season
            lines.append("")
            lines.append("-- Constructor total points by race in 2023")
            lines.append("SELECT r.name AS grand_prix, c.name AS constructor, SUM(cr.points) AS points")
            lines.append("FROM constructor_results cr")
            lines.append("JOIN races r ON r.raceId = cr.raceId")
            lines.append("JOIN constructors c ON c.constructorId = cr.constructorId")
            lines.append("WHERE r.year = 2023")
            lines.append("GROUP BY r.raceId, c.constructorId")
            lines.append("ORDER BY r.round, points DESC")
            lines.append("LIMIT 50")
        return "\n".join(lines)
    
    @staticmethod
    def _call_ollama(prompt: str) -> Optional[str]:
        """Call Ollama API for LLM generation."""
        try:
            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {"temperature": 0},
                    "stream": False
                },
                timeout=config.LLM_TIMEOUT
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract content from Ollama response
            content = (data.get("message") or {}).get("content", "") or ""
            return content.strip()
            
        except requests.RequestException as e:
            print(f"Ollama API error: {e}")
            return None
    
    @staticmethod
    def _call_groq(prompt: str) -> Optional[str]:
        """Call Groq API for LLM generation with basic 429-retry backoff."""
        client = LLMService._get_groq_client()
        if not client:
            print("Groq client not available (missing OpenAI package or API key)")
            return None
        import time
        attempts = 3
        backoff = 2
        for i in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,  # More deterministic for SQL generation
                    max_tokens=1000,
                    timeout=config.LLM_TIMEOUT
                )
                content = response.choices[0].message.content
                return content.strip() if content else None
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    sleep_for = backoff ** i
                    print(f"Groq rate limit hit, retrying in {sleep_for}s (attempt {i+1}/{attempts})")
                    time.sleep(sleep_for)
                    continue
                print(f"Groq API error: {e}")
                return None
        return None
    
    @staticmethod
    def _repair_sql(question: str, schema: str, original_sql: str, error_message: str) -> Optional[str]:
        """Ask the LLM to repair SQL based on a SQLite error, returning a corrected SELECT."""
        repair_prompt = (
            "You previously generated a SQLite SELECT query that caused an error. "
            "Revise the SQL to fix the error while preserving the user's intent. "
            "Return ONLY a valid SQLite SELECT statement. No comments or markdown.\n\n"
            f"Error: {error_message}\n\n"
            f"Original SQL:\n{original_sql}\n\n"
            f"Question: {question}\n"
            "SQL:"
        )
        # Choose backend based on configuration
        if config.LLM_BACKEND == "groq":
            return LLMService._call_groq(repair_prompt)
        else:
            return LLMService._call_ollama(repair_prompt)

    @staticmethod
    def get_sql_query(question: str, schema: str, compact_schema: Optional[str] = None) -> Optional[str]:
        """
        Generate a SQL query from a natural language question.
        
        Args:
            question: Natural language question
            schema: Database schema description
            
        Returns:
            SQL query string or None if generation failed
        """
        if not question or not schema:
            return None
            
        # Optionally use RAG to reduce schema context
        rag_schema = None
        # If a compact_schema is provided by the caller (e.g., /askdb pre-retrieval), use it
        if compact_schema:
            rag_schema = compact_schema
        elif getattr(config, 'USE_RAG', False):
            try:
                from rag.retriever import SchemaRetriever, build_context_block
                retriever = SchemaRetriever()
                retrieved = retriever.retrieve_relevant_schema(question)
                compact = build_context_block(retrieved)
                if compact:
                    print(f"[RAG] Retrieved tables: {retrieved.get('tables')}")
                    print(f"[RAG] Retrieved columns: {retrieved.get('columns')}")
                    rag_schema = compact
            except Exception as e:
                print(f"[RAG] Retrieval failed, falling back to full schema: {e}")
                rag_schema = None
        prompt = LLMService.generate_prompt(question, rag_schema or schema)
        
        # Choose backend based on configuration
        if config.LLM_BACKEND == "groq":
            content = LLMService._call_groq(prompt)
        else:  # Default to Ollama
            content = LLMService._call_ollama(prompt)
        
        if not content:
            return None
        
        try:
            # Clean up the response
            content = content.strip().strip("`")
            content = re.sub(r"^sql\n", "", content, flags=re.IGNORECASE).strip()
            content = content.rstrip(";").strip()
            
            # Validate it's a SELECT statement
            if not re.match(r"^\s*(WITH\s+.+?\s+)?SELECT\b", content, re.IGNORECASE | re.DOTALL):
                print(f"LLM generated non-SELECT query: {content}")
                return None

            # If a constructor/team is in question but SQL lacks constructor filter, repair pre-EXPLAIN
            team_kw = LLMService._extract_constructor_keyword(question)
            if team_kw and not LLMService._sql_has_constructor_filter(content, team_kw):
                guidance = (
                    "The query must filter to the specified constructor/team by joining through results to constructors "
                    f"and applying a filter on constructors.name or constructorRef that matches '{team_kw}'."
                )
                repair_prompt = (
                    "Revise the following SQLite SELECT to enforce the constructor/team filter. "
                    "Return ONLY a valid SQLite SELECT. No comments or markdown.\n\n"
                    f"Guidance: {guidance}\n\nOriginal SQL:\n{content}\n\nQuestion: {question}\nSQL:"
                )
                content2 = LLMService._call_groq(repair_prompt) if config.LLM_BACKEND == "groq" else LLMService._call_ollama(repair_prompt)
                if content2:
                    content2 = content2.strip().strip('`').rstrip(';').strip()
                    if re.match(r"^\s*(WITH\s+.+?\s+)?SELECT\b", content2, re.IGNORECASE | re.DOTALL):
                        content = content2
            
            # Apply guardrails
            is_safe, safe_sql, reason = sanitize_sql(content, config.DEFAULT_LIMIT)
            if not is_safe:
                print(f"Guardrails rejected SQL: {reason} - {content}")
                return None

            # Judge with EXPLAIN; attempt single repair on error
            preflight = DatabaseService.explain_query(safe_sql)
            if 'error' in preflight:
                print(f"[SQL Preflight] EXPLAIN error: {preflight['error']}. Attempting repair...")
                repaired = LLMService._repair_sql(question, schema, safe_sql, preflight['error'])
                if repaired:
                    # Clean and sanitize repaired SQL
                    repaired = repaired.strip().strip('`').rstrip(';').strip()
                    if re.match(r"^\s*(WITH\s+.+?\s+)?SELECT\b", repaired, re.IGNORECASE | re.DOTALL):
                        ok, safe_repaired, reason2 = sanitize_sql(repaired, config.DEFAULT_LIMIT)
                        if ok:
                            preflight2 = DatabaseService.explain_query(safe_repaired)
                            if 'ok' in preflight2:
                                return safe_repaired
                            else:
                                print(f"[SQL Preflight] Repaired EXPLAIN failed: {preflight2['error']}")
                        else:
                            print(f"[Repair] Guardrails rejected repaired SQL: {reason2}")
                # If repair failed, fall back to original if preflight allowed? Otherwise None
                return None

            return safe_sql
            
        except Exception as e:
            print(f"Unexpected error in LLM service: {e}")
            return None
    
    @staticmethod
    def generate_insights(question: str, sql_query: str, data_results: list, schema: str, custom_prompt: str = None) -> Optional[str]:
        """
        Generate data-driven insights from query results.
        
        Args:
            question: Original natural language question
            sql_query: The SQL query that was executed
            data_results: Query results data
            schema: Database schema description
            
        Returns:
            Generated insights text or None if generation failed
        """
        if not question or not sql_query or not data_results:
            return None
        
        # Limit data for prompt (take first 10 rows to avoid token limits)
        sample_data = data_results[:10] if len(data_results) > 10 else data_results
        
        # Format data for the prompt
        data_summary = f"Total rows returned: {len(data_results)}\n"
        if sample_data:
            data_summary += f"Sample data (first {len(sample_data)} rows):\n"
            for i, row in enumerate(sample_data, 1):
                data_summary += f"Row {i}: {dict(row)}\n"
        
        # Use custom prompt if provided, otherwise use default insights prompt
        if custom_prompt:
            insights_prompt = custom_prompt
        else:
            insights_prompt = (
                "You are a data analyst providing insights on query results for a Slack message. "
                "Analyze the data and provide meaningful, actionable insights based on the original question and results. "
                "Focus on trends, patterns, outliers, and business implications. "
                "Keep your response concise but insightful (2-4 paragraphs).\n\n"
                "FORMATTING GUIDELINES:\n"
                "- Use **bold** for key findings and important metrics\n"
                "- Use bullet points (- or •) for lists\n"
                "- Use clear section headers with **bold** formatting\n"
                "- Keep sentences concise and scannable\n"
                "- Avoid excessive nested formatting\n\n"
                f"Original Question: {question}\n"
                f"SQL Query: {sql_query}\n"
                f"Database Schema Context:\n{schema[:500]}...\n\n"  # Limit schema for token efficiency
                f"Query Results Summary:\n{data_summary}\n\n"
                "Please provide data-driven insights with clear formatting:"
            )
        
        # Choose backend based on configuration
        if config.LLM_BACKEND == "groq":
            insights = LLMService._call_groq(insights_prompt)
        else:  # Default to Ollama
            insights = LLMService._call_ollama(insights_prompt)
        
        if not insights:
            print("LLM returned empty insights")
            return None
            
        return insights
    
    @staticmethod
    def generate_text_response(prompt: str) -> Optional[str]:
        """
        Generate a text response using the LLM for custom prompts.
        
        Args:
            prompt: Custom prompt for text generation
            
        Returns:
            Generated text or None if generation failed
        """
        if not prompt:
            return None
        
        # Choose backend based on configuration
        if config.LLM_BACKEND == "groq":
            # Use the existing OpenAI-compatible Groq client (no tools)
            try:
                client = LLMService._get_groq_client()
                if not client:
                    print("Groq client not available")
                    return None
                
                response = client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1500
                    # Remove tool_choice to avoid the error
                )
                
                content = response.choices[0].message.content
                return content.strip() if content else None
                
            except Exception as e:
                print(f"Groq text generation error: {e}")
                return None
        else:  # Default to Ollama
            response = LLMService._call_ollama(prompt)
            return response