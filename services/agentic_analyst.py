"""
Agentic database analyst service for comprehensive question answering.
"""
import json
import re
from typing import List, Dict, Optional, Tuple
from services.llm import LLMService
from services.database import DatabaseService
from config.settings import config


class AgenticAnalyst:
    """
    An AI-powered database analyst that can investigate business questions
    by planning and executing multiple SQL queries, then synthesizing insights.
    """
    
    @staticmethod
    def investigate_question(question: str, schema: str) -> str:
        """
        Investigate a business question using an agentic approach.
        
        Args:
            question: Business question to investigate
            schema: Database schema description
            
        Returns:
            Comprehensive analysis as formatted text
        """
        try:
            print(f"[ASKDB] Starting investigation: {question}")
            
            # Step 1: Analyze the question and create investigation plan
            investigation_plan = AgenticAnalyst._plan_investigation(question, schema)
            if not investigation_plan:
                return "❌ I couldn't understand how to investigate that question. Please try rephrasing it."
            
            print(f"[ASKDB] Investigation plan: {investigation_plan}")
            
            # Step 2: Generate SQL queries based on the plan
            queries = AgenticAnalyst._generate_investigation_queries(investigation_plan, schema)
            if not queries:
                print(f"[ASKDB] Query generation failed, trying fallback")
                # Create a simple fallback query based on question type
                if "driver" in question.lower() and ("won" in question.lower() or "champion" in question.lower()):
                    year = "2021" if "2021" in question else "2020" if "2020" in question else "2023"
                    queries = [f"SELECT d.forename, d.surname, SUM(r.points) as total_points FROM drivers d JOIN results r ON d.driverId = r.driverId JOIN races ra ON r.raceId = ra.raceId WHERE ra.year = {year} GROUP BY d.driverId ORDER BY total_points DESC LIMIT 5"]
                elif "driver" in question.lower() and ("most races" in question.lower() or "participated" in question.lower()):
                    queries = [
                        "SELECT d.forename, d.surname, COUNT(r.raceId) as race_count FROM drivers d JOIN results r ON d.driverId = r.driverId GROUP BY d.driverId ORDER BY race_count DESC LIMIT 10",
                        "SELECT COUNT(DISTINCT raceId) as total_races FROM results"
                    ]
                elif "team" in question.lower() or "constructor" in question.lower():
                    year = "2020" if "2020" in question else "2021" if "2021" in question else "2023"
                    queries = [f"SELECT c.name, SUM(cr.points) as total_points FROM constructors c JOIN constructor_results cr ON c.constructorId = cr.constructorId JOIN races r ON cr.raceId = r.raceId WHERE r.year = {year} GROUP BY c.constructorId ORDER BY total_points DESC LIMIT 5"]
                else:
                    # Generic fallback for dataset overview
                    queries = [
                        "SELECT COUNT(*) as total_drivers FROM drivers",
                        "SELECT COUNT(*) as total_races FROM races",
                        "SELECT MIN(year) as first_year, MAX(year) as last_year FROM races"
                    ]
            
            print(f"[ASKDB] Generated {len(queries)} queries")
            
            # Debug: Show all generated queries
            for i, query in enumerate(queries):
                print(f"[ASKDB] Query {i+1}: {query}")
            
            # Step 3: Execute queries and collect results
            query_results = []
            for i, query in enumerate(queries):
                print(f"[ASKDB] Executing query {i+1}/{len(queries)}: {query[:100]}...")
                result = AgenticAnalyst._execute_investigation_query(query)
                print(f"[ASKDB] Query {i+1} result: {len(result) if result else 0} rows")
                if result:
                    query_results.append({
                        'query': query,
                        'data': result,
                        'description': f"Query {i+1}"
                    })
            
            if not query_results:
                return "❌ I couldn't retrieve any data for this investigation. Please check your question."
            
            print(f"[ASKDB] Collected {len(query_results)} successful results")
            
            # Debug: Show what data we collected
            try:
                for i, result in enumerate(query_results):
                    print(f"[ASKDB] Result {i+1}: {len(result.get('data', [])) if result.get('data') else 0} rows")
                    if result.get('data') and len(result['data']) > 0:
                        print(f"[ASKDB] Sample data: {result['data'][0]}")
                    else:
                        print(f"[ASKDB] No data in result {i+1}")
            except Exception as debug_error:
                print(f"[ASKDB] Debug error: {debug_error}")
            
            # Step 4: Synthesize insights from all results
            print(f"[ASKDB] Starting synthesis step...")
            try:
                comprehensive_analysis = AgenticAnalyst._synthesize_insights(
                    question, investigation_plan, query_results, schema
                )
                print(f"[ASKDB] Synthesis completed, result length: {len(comprehensive_analysis) if comprehensive_analysis else 0}")
            except Exception as synthesis_error:
                print(f"[ASKDB] Synthesis failed with exception: {synthesis_error}")
                import traceback
                traceback.print_exc()
                comprehensive_analysis = AgenticAnalyst._create_fallback_response(question, query_results)
            
            if not comprehensive_analysis:
                print(f"[ASKDB] Comprehensive analysis is empty or None")
                return "❌ I collected data but couldn't generate meaningful insights. Please try a different question."
            
            print(f"[ASKDB] Generated comprehensive analysis: {comprehensive_analysis[:200]}...")
            return comprehensive_analysis
            
        except Exception as e:
            print(f"[ASKDB] Error in investigation: {e}")
            return f"❌ I encountered an error during investigation: {str(e)}"
    
    @staticmethod
    def _plan_investigation(question: str, schema: str) -> Optional[Dict]:
        """Create an investigation plan for the given question."""
        prompt = f"""
You are a data analyst planning how to investigate a business question.

Question: {question}
Database Schema: {schema}

Create an investigation plan. Respond with ONLY a JSON object containing:
{{
    "intent": "what the user wants to understand (trend_analysis, comparison, root_cause, performance_review, etc.)",
    "key_metrics": ["list of metrics to analyze (revenue, count, average, etc.)"],
    "dimensions": ["list of dimensions to break down by (time, product, region, customer_type, etc.)"],
    "time_scope": "relevant time period (last_month, this_year, last_30_days, etc.)",
    "investigation_type": "type of analysis needed (trend, comparison, breakdown, correlation, etc.)",
    "focus_areas": ["specific areas to investigate based on the question"]
}}

Examples:
- "why did sales drop?" → focus on time trends, comparisons, potential causes
- "who are my best customers?" → focus on customer segmentation, ranking, metrics
- "which products are profitable?" → focus on product performance, profitability metrics

Respond with ONLY the JSON object.
"""
        
        try:
            if config.LLM_BACKEND == "groq":
                response = LLMService._call_groq(prompt)
            else:
                response = LLMService._call_ollama(prompt)
            
            if not response:
                return None
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return None
            
        except Exception as e:
            print(f"Error planning investigation: {e}")
            return None
    
    @staticmethod
    def _generate_investigation_queries(plan: Dict, schema: str) -> List[str]:
        """Generate SQL queries based on the investigation plan."""
        prompt = f"""
You are a SQL expert generating queries for a business investigation.

Investigation Plan: {json.dumps(plan, indent=2)}
Database Schema: {schema}

Generate 3-5 simple SQLite SELECT queries. Each query should be SHORT and focused:

Examples for dataset overview:
SELECT COUNT(*) as total_drivers FROM drivers
SELECT COUNT(*) as total_races FROM races  
SELECT MIN(year) as first_year, MAX(year) as last_year FROM races
SELECT COUNT(DISTINCT nationality) as countries FROM drivers
SELECT name FROM constructors ORDER BY name LIMIT 10

Rules:
- Keep each query SIMPLE (avoid complex subqueries)
- One main metric per query
- Use basic COUNT, MIN, MAX, AVG functions
- Each query should be under 100 characters
- Only SELECT statements
- Use SQLite syntax

Respond with ONLY the SQL queries, one per line, no explanations.
"""
        
        try:
            if config.LLM_BACKEND == "groq":
                response = LLMService._call_groq(prompt)
            else:
                response = LLMService._call_ollama(prompt)
            
            print(f"[ASKDB] Query generation response: {response}")
            
            if not response or response.strip() == "" or response.lower() == "none":
                print(f"[ASKDB] Empty/None response from query generation")
                return []
            
            # Extract SQL queries from response
            queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line and line.upper().startswith('SELECT'):
                    # Clean up the query
                    query = line.rstrip(';').strip()
                    queries.append(query)
            
            print(f"[ASKDB] Extracted {len(queries)} queries from response")
            
            if len(queries) == 0:
                print(f"[ASKDB] No SELECT statements found in response")
                
            return queries[:5]  # Limit to 5 queries max
            
        except Exception as e:
            print(f"Error generating queries: {e}")
            return []
    
    @staticmethod
    def _execute_investigation_query(query: str) -> Optional[List[Dict]]:
        """Execute a query safely and return results."""
        try:
            # Apply same guardrails as regular queries
            from core.guardrails import sanitize_sql
            is_safe, safe_query, reason = sanitize_sql(query, 100)  # Limit to 100 rows for investigation
            
            if not is_safe:
                print(f"Query failed guardrails: {reason}")
                return None
            
            # Execute query - DatabaseService returns {'data': [...]} so extract the data
            result = DatabaseService.execute_query(safe_query)
            if isinstance(result, dict) and 'data' in result:
                return result['data']
            else:
                print(f"Unexpected result format: {result}")
                return None
            
        except Exception as e:
            print(f"Error executing investigation query: {e}")
            return None
    
    @staticmethod
    def _synthesize_insights(question: str, plan: Dict, results: List[Dict], schema: str) -> str:
        """Synthesize all query results into comprehensive business insights."""
        # Format results for LLM analysis
        formatted_results = ""
        for i, result in enumerate(results):
            formatted_results += f"\n--- Query {i+1} Results ---\n"
            formatted_results += f"SQL: {result['query'][:200]}{'...' if len(result['query']) > 200 else ''}\n"
            
            # Safely get data length
            data = result.get('data', [])
            data_len = len(data) if isinstance(data, list) else 0
            formatted_results += f"Data ({data_len} rows):\n"
            
            # Handle different data formats safely
            data = result.get('data', [])
            if data and isinstance(data, list) and len(data) > 0:
                # Show first few rows with column headers
                first_row = data[0]
                if isinstance(first_row, dict):
                    headers = list(first_row.keys())
                    formatted_results += f"Columns: {', '.join(headers)}\n"
                    
                    # Limit to 5 rows for shorter prompts
                    for row in data[:5]:  
                        if isinstance(row, dict):
                            row_str = ", ".join([f"{k}: {str(v)[:50]}{'...' if len(str(v)) > 50 else ''}" for k, v in row.items()])
                            formatted_results += f"  {row_str}\n"
                    
                    if len(data) > 5:
                        formatted_results += f"  ... ({len(data) - 5} more rows)\n"
                else:
                    formatted_results += f"  Data: {str(data)[:200]}...\n"
            else:
                formatted_results += "  No data returned\n"
        
        # Check if prompt will be too long
        estimated_prompt_length = len(formatted_results) + 1500  # Base prompt + plan
        print(f"[ASKDB] Estimated prompt length: {estimated_prompt_length} chars")
        
        # Create a more focused prompt if data is complex
        if estimated_prompt_length > 8000:  # Groq limit is around 8k tokens
            formatted_results = f"Data Summary: {len(results)} queries executed with results ranging from {min(len(r['data']) for r in results)} to {max(len(r['data']) for r in results)} rows each."
        
        prompt = f"""
You are a business analyst. A user asked: "{question}"

Data collected from {len(results)} queries:
{formatted_results}

Provide a concise analysis in exactly this format (KEEP IT SHORT - max 600 characters):

🤖 *{question.title()}*

📊 *Key Findings:*
• [Top insight with numbers]
• [Second key insight]
• [Third insight]

💡 *Summary:*
[1-2 sentences explaining what this means for the business]

*Based on {len(results)} database queries*

CRITICAL: 
- Maximum 1200 characters total
- Use specific numbers from the data
- Be concise and actionable
- Answer the specific question asked
- If asking "who won", provide the actual winner's name
"""
        
        try:
            print(f"[ASKDB] Synthesizing insights with prompt length: {len(prompt)}")
            print(f"[ASKDB] Prompt preview: {prompt[:500]}...")
            
            if config.LLM_BACKEND == "groq":
                response = LLMService._call_groq(prompt)
            else:
                response = LLMService._call_ollama(prompt)
            
            print(f"[ASKDB] Raw LLM response: {response}")
            print(f"[ASKDB] Synthesis response received: {len(response) if response else 0} characters")
            
            if not response:
                print(f"[ASKDB] Empty response from LLM during synthesis - using fallback")
                # Create a simple fallback response with the data
                fallback = AgenticAnalyst._create_fallback_response(question, results)
                print(f"[ASKDB] Fallback response: {fallback}")
                return fallback
            
            print(f"[ASKDB] Returning LLM response: {response[:200]}...")
            return response
            
        except Exception as e:
            print(f"[ASKDB] Exception during synthesis: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to provide a fallback response
            try:
                print(f"[ASKDB] Attempting fallback response...")
                fallback = AgenticAnalyst._create_fallback_response(question, results)
                print(f"[ASKDB] Fallback created: {fallback[:200]}...")
                return fallback
            except Exception as fallback_error:
                print(f"[ASKDB] Fallback also failed: {fallback_error}")
                return f"❌ Failed to generate insights due to an error: {str(e)}"
    
    @staticmethod
    def _create_fallback_response(question: str, results: List[Dict]) -> str:
        """Create a simple fallback response when LLM synthesis fails."""
        response = f"🤖 **Data Investigation: {question.title()}**\n\n"
        response += f"📊 **Data Summary:**\n"
        response += f"• Executed {len(results)} database queries\n"
        
        total_rows = sum(len(r['data']) for r in results)
        response += f"• Retrieved {total_rows} total data points\n\n"
        
        response += "📋 **Query Results:**\n"
        for i, result in enumerate(results):
            response += f"• Query {i+1}: {len(result['data'])} rows returned\n"
            if result['data'] and len(result['data']) > 0:
                # Show a sample of the first result
                sample = result['data'][0]
                keys = list(sample.keys())[:3]  # Show first 3 columns
                sample_data = ", ".join([f"{k}: {sample[k]}" for k in keys])
                response += f"  Sample: {sample_data}\n"
        
        response += f"\n💡 **Next Steps:**\n"
        response += f"• Try rephrasing your question for better analysis\n"
        response += f"• Use `/dd` command for specific data queries\n"
        response += f"• Contact support if this persists\n"
        
        response += f"\n*Investigation completed with {len(results)} successful queries*"
        return response