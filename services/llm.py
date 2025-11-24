"""
LLM service for generating SQL queries using Ollama.
"""
import requests
import re
from typing import Optional
from config.settings import config
from core.guardrails import sanitize_sql


class LLMService:
    """Handles LLM operations for SQL generation."""
    
    @staticmethod
    def generate_prompt(question: str, schema: str) -> str:
        """
        Generate a prompt for the LLM to create SQL queries.
        
        Args:
            question: Natural language question
            schema: Database schema description
            
        Returns:
            Formatted prompt string
        """
        return (
            "You are a strict SQLite SQL generator. "
            "Always respond with ONLY a valid SQLite SELECT statement (no explanations, no markdown fences). "
            "Use the provided schema. Avoid DDL/DML; only read data. "
            f"\n\nSCHEMA:\n{schema}\n\n"
            f"Question: {question}\n"
            "SQL:"
        )
    
    @staticmethod
    def get_sql_query(question: str, schema: str) -> Optional[str]:
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
            
        prompt = LLMService.generate_prompt(question, schema)
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False
                },
                timeout=config.LLM_TIMEOUT
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract content from Ollama response
            content = (data.get("message") or {}).get("content", "") or ""
            
            # Clean up the response
            content = content.strip().strip("`")
            content = re.sub(r"^sql\n", "", content, flags=re.IGNORECASE).strip()
            content = content.rstrip(";").strip()
            
            # Validate it's a SELECT statement
            if not re.match(r"^\s*(WITH\s+.+?\s+)?SELECT\b", content, re.IGNORECASE | re.DOTALL):
                print(f"LLM generated non-SELECT query: {content}")
                return None
            
            # Apply guardrails
            is_safe, safe_sql, reason = sanitize_sql(content, config.DEFAULT_LIMIT)
            if not is_safe:
                print(f"Guardrails rejected SQL: {reason} - {content}")
                return None
                
            return safe_sql
            
        except requests.RequestException as e:
            print(f"Ollama API error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in LLM service: {e}")
            return None
    
    @staticmethod
    def generate_insights(question: str, sql_query: str, data_results: list, schema: str) -> Optional[str]:
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
        
        insights_prompt = (
            "You are a data analyst providing insights on query results. "
            "Analyze the data and provide meaningful, actionable insights based on the original question and results. "
            "Focus on trends, patterns, outliers, and business implications. "
            "Keep your response concise but insightful (2-4 paragraphs).\n\n"
            f"Original Question: {question}\n"
            f"SQL Query: {sql_query}\n"
            f"Database Schema Context:\n{schema[:500]}...\n\n"  # Limit schema for token efficiency
            f"Query Results Summary:\n{data_summary}\n\n"
            "Please provide data-driven insights:"
        )
        
        try:
            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are an expert data analyst who provides clear, actionable insights from data."},
                        {"role": "user", "content": insights_prompt},
                    ],
                    "stream": False
                },
                timeout=config.LLM_TIMEOUT
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract insights content
            insights = (data.get("message") or {}).get("content", "") or ""
            insights = insights.strip()
            
            if not insights:
                print("LLM returned empty insights")
                return None
                
            return insights
            
        except requests.RequestException as e:
            print(f"Ollama API error for insights: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in insights generation: {e}")
            return None