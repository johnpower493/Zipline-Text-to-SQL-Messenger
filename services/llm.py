"""
LLM service for generating SQL queries using Ollama or Groq.
"""
import requests
import re
from typing import Optional
from config.settings import config
from core.guardrails import sanitize_sql

# Optional import for Groq support
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMService:
    """Handles LLM operations for SQL generation."""
    
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
        """Call Groq API for LLM generation."""
        try:
            client = LLMService._get_groq_client()
            if not client:
                print("Groq client not available (missing OpenAI package or API key)")
                return None
            
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
            print(f"Groq API error: {e}")
            return None
    
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
            
            # Apply guardrails
            is_safe, safe_sql, reason = sanitize_sql(content, config.DEFAULT_LIMIT)
            if not is_safe:
                print(f"Guardrails rejected SQL: {reason} - {content}")
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