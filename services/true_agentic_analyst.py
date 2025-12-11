"""
True Agentic Database Analyst - Iterative reasoning and dynamic investigation.

This implements a proper agentic system that:
- Reasons through problems step by step
- Adapts strategy based on findings
- Iteratively refines its approach
- Makes intelligent decisions about next steps
"""
import json
import time
from typing import List, Dict, Optional, Any
from services.database import DatabaseService
from services.llm import LLMService
from config.settings import config


class TrueAgenticAnalyst:
    """
    A true agentic database analyst that iteratively investigates questions
    using dynamic reasoning and adaptive query generation.
    """
    
    def __init__(self):
        # Reduce iterations if RAG is enabled to mitigate rate limits
        self.max_iterations = getattr(config, 'AGENT_MAX_ITER_RAG', 3) if getattr(config, 'USE_RAG', False) else 5
        self.max_duration_seconds = getattr(config, 'LLM_TIMEOUT', 120) * 2  # hard wall-clock cap
        self.findings_history = []
        self.reasoning_chain = []
    
    def investigate(self, question: str, schema: str) -> str:
        """
        Main agentic investigation with iterative reasoning.
        
        Args:
            question: Business question to investigate
            schema: Database schema description
            
        Returns:
            Comprehensive analysis based on iterative findings
        """
        try:
            print(f"[AGENTIC] Starting true agentic investigation: {question}")
            start_time = time.time()
            
            # Clear previous investigation state
            self.findings_history = []
            self.reasoning_chain = []
            
            # Step 1: Initial reasoning about the question
            print(f"[AGENTIC] Step 1: Reasoning about the question...")
            initial_reasoning = self._reason_about_question(question, schema)
            if not initial_reasoning:
                return "❌ I couldn't understand how to approach this question."
            
            print(f"[AGENTIC] Initial reasoning: {initial_reasoning}")
            self.reasoning_chain.append({"step": "initial_reasoning", "content": initial_reasoning})
            
            # Step 2: Form hypothesis and generate first query
            print(f"[AGENTIC] Step 2: Forming hypothesis...")
            hypothesis = self._form_initial_hypothesis(question, initial_reasoning, schema)
            if not hypothesis:
                return "❌ I couldn't form a hypothesis for this investigation."
            
            print(f"[AGENTIC] Hypothesis: {hypothesis}")
            self.reasoning_chain.append({"step": "hypothesis", "content": hypothesis})
            
            # Step 3: Iterative investigation
            print(f"[AGENTIC] Step 3: Starting iterative investigation...")
            current_focus = hypothesis.get("initial_query_focus", "")

            # RAG once per investigation: build compact schema once and reuse
            compact_schema = None
            if getattr(config, 'USE_RAG', False):
                try:
                    from rag.retriever import SchemaRetriever, build_context_block
                    retriever = SchemaRetriever()
                    retrieved = retriever.retrieve_relevant_schema(question)
                    compact_schema = build_context_block(retrieved)
                    if compact_schema:
                        print(f"[RAG] (askdb) Reusing compact schema. Tables: {retrieved.get('tables')}")
                except Exception as e:
                    print(f"[RAG] (askdb) Pre-retrieval failed: {e}")
                    compact_schema = None
            
            for iteration in range(self.max_iterations):
                # Hard wall-clock timeout to prevent Slack hangs
                if time.time() - start_time > self.max_duration_seconds:
                    print(f"[AGENTIC] Max duration reached, synthesizing partial answer")
                    break
                print(f"[AGENTIC] --- Iteration {iteration + 1}/{self.max_iterations} ---")
                
                # Optional pacing delay to reduce rate-limit bursts
                try:
                    time.sleep(getattr(config, 'AGENT_ITERATION_DELAY', 1.0))
                except Exception:
                    pass
                
                # Generate next query based on current understanding
                next_query = self._generate_focused_query(
                    current_focus, question, schema, iteration, compact_schema=compact_schema
                )
                
                if not next_query:
                    print(f"[AGENTIC] No query generated, ending investigation")
                    break
                
                print(f"[AGENTIC] Query: {next_query}")
                
                # Execute query and get results
                result = self._execute_safe_query(next_query)
                if result:
                    self.findings_history.append({
                        "iteration": iteration + 1,
                        "query": next_query,
                        "data": result,
                        "focus": current_focus
                    })
                    print(f"[AGENTIC] Found {len(result)} rows of data")
                
                # Analyze findings and decide next step
                analysis = self._analyze_current_findings(question, hypothesis)
                print(f"[AGENTIC] Analysis: {analysis.get('status', 'unknown')}")
                
                # Decide what to do next
                next_action = self._decide_next_action(analysis, question, iteration)
                print(f"[AGENTIC] Next action: {next_action.get('action', 'unknown')}")
                
                if next_action.get("action") == "complete":
                    print(f"[AGENTIC] Investigation complete!")
                    break
                elif next_action.get("action") == "continue":
                    current_focus = next_action.get("new_focus", current_focus)
                elif next_action.get("action") == "pivot":
                    current_focus = next_action.get("pivot_focus", current_focus)
                    print(f"[AGENTIC] Pivoting investigation focus to: {current_focus}")
            
            # Step 4: Synthesize final answer
            print(f"[AGENTIC] Step 4: Synthesizing final answer...")
            final_answer = self._synthesize_final_answer(question, hypothesis)
            
            total_time = time.time() - start_time
            print(f"[AGENTIC] Investigation completed in {total_time:.2f}s with {len(self.findings_history)} findings")
            
            return final_answer
            
        except Exception as e:
            print(f"[AGENTIC] Error in agentic investigation: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Investigation failed: {str(e)}"
    
    def _reason_about_question(self, question: str, schema: str) -> Optional[Dict]:
        """Step 1: Reason about what the question is asking."""
        prompt = f"""
You are reasoning about how to investigate a business question.

Question: "{question}"
Database Schema: {schema[:1000]}...

Think step by step about this question:
1. What is the user really asking for?
2. What type of data do I need to answer this?
3. What are the key entities involved?
4. What's my investigation strategy?

Respond with ONLY a JSON object:
{{
    "question_type": "ranking|comparison|trend|overview|specific_fact",
    "main_entities": ["entity1", "entity2"],
    "data_needed": ["what data to find"],
    "investigation_strategy": "brief strategy description",
    "complexity": "simple|medium|complex"
}}
"""
        
        return self._call_llm_for_json(prompt, "reasoning")
    
    def _form_initial_hypothesis(self, question: str, reasoning: Dict, schema: str) -> Optional[Dict]:
        """Step 2: Form a testable hypothesis about the answer."""
        prompt = f"""
Based on this reasoning about a question, form an investigation hypothesis.

Question: "{question}"
Reasoning: {json.dumps(reasoning, indent=2)}

Create an investigation plan with ONLY this JSON format:
{{
    "hypothesis": "what I think the answer might be",
    "initial_query_focus": "what to investigate first",
    "expected_findings": "what I expect to discover",
    "verification_needed": "how to verify the answer",
    "backup_approach": "alternative if first approach fails"
}}
"""
        
        return self._call_llm_for_json(prompt, "hypothesis")
    
    def _generate_focused_query(self, focus: str, question: str, schema: str, iteration: int, compact_schema: Optional[str] = None) -> Optional[str]:
        """Generate a focused query using the same proven method as /dd command."""
        from services.llm import LLMService
        
        # Use the SAME SQL generation approach as /dd command
        print(f"[AGENTIC] Using /dd-style SQL generation for reliability...")
        
        # Create focused question for this iteration
        if iteration == 0:
            # First iteration: direct question
            focused_question = question
        else:
            # Later iterations: build on findings
            if self.findings_history:
                latest = self.findings_history[-1]
                if len(latest["data"]) > 0:
                    # We have data, now dig deeper based on what we found
                    first_result = latest["data"][0]
                    if isinstance(first_result, dict) and len(first_result) > 0:
                        # Use actual data to inform next question
                        key_field = list(first_result.keys())[0]
                        value = list(first_result.values())[0]
                        focused_question = f"detailed analysis of {key_field} data patterns"
                    else:
                        focused_question = f"explore {focus} in more depth"
                else:
                    # No data, try different angle
                    focused_question = f"alternative approach to {focus}"
            else:
                focused_question = question
        
        print(f"[AGENTIC] Focused question for iteration {iteration + 1}: {focused_question}")
        
        # Use the PROVEN /dd SQL generation method
        sql_query = LLMService.get_sql_query(focused_question, schema, compact_schema=compact_schema)
        
        if sql_query:
            print(f"[AGENTIC] /dd-style generation succeeded: {sql_query}")
            return sql_query
        else:
            print(f"[AGENTIC] /dd-style generation failed, using intelligent fallback")
            
            # Completely database-agnostic smart fallbacks
            return self._generate_schema_aware_fallback(question, focus, iteration, schema)
    
    def _analyze_current_findings(self, question: str, hypothesis: Dict) -> Dict:
        """Analyze current findings and determine if we need more investigation."""
        if not self.findings_history:
            return {"status": "no_data", "confidence": 0}
        
        latest_finding = self.findings_history[-1]
        total_findings = len(self.findings_history)
        data_count = len(latest_finding["data"])
        
        print(f"[AGENTIC] Analyzing: {data_count} rows in iteration {total_findings}")
        
        if data_count == 0:
            return {
                "status": "no_data_found",
                "confidence": 0,
                "insight": "No data returned from last query",
                "next_focus": "try_different_approach"
            }
        
        # Be more curious - don't stop at first successful query
        if total_findings == 1:
            # First iteration success - always dig deeper
            return {
                "status": "initial_data_found",
                "confidence": 40,  # Deliberately low to encourage more investigation
                "insight": f"Found initial data ({data_count} rows), but need deeper analysis",
                "next_focus": "explore_details"
            }
        elif total_findings == 2:
            # Second iteration - look for patterns or validation
            return {
                "status": "pattern_emerging",
                "confidence": 65,
                "insight": f"Have {total_findings} data points, looking for patterns or validation",
                "next_focus": "validate_findings"
            }
        elif total_findings >= 3:
            # Multiple findings - high confidence if data is consistent
            return {
                "status": "comprehensive_data",
                "confidence": 85,
                "insight": f"Have {total_findings} comprehensive findings",
                "next_focus": "final_verification"
            }
        else:
            return {
                "status": "need_more_context",
                "confidence": 50,
                "insight": "Need additional context to form complete understanding",
                "next_focus": "gather_context"
            }
    
    def _decide_next_action(self, analysis: Dict, question: str, iteration: int) -> Dict:
        """Decide what to do next based on current analysis - database agnostic."""
        status = analysis.get("status", "unknown")
        confidence = analysis.get("confidence", 0)
        next_focus = analysis.get("next_focus", "")
        
        print(f"[AGENTIC] Decision point: status={status}, confidence={confidence}, iteration={iteration}")
        
        # Max iterations check
        if iteration >= self.max_iterations - 1:
            return {"action": "complete", "reason": "max_iterations_reached"}
        
        # Only complete if we have high confidence AND multiple findings
        if confidence >= 85 and len(self.findings_history) >= 2:
            return {"action": "complete", "reason": "high_confidence_with_multiple_findings"}
        
        # Be more persistent - explore the data more deeply
        if status == "initial_data_found":
            return {
                "action": "continue",
                "new_focus": self._generate_deeper_focus(question),
                "reason": "explore_initial_findings_deeper"
            }
        
        elif status == "pattern_emerging":
            return {
                "action": "continue", 
                "new_focus": self._generate_validation_focus(question),
                "reason": "validate_patterns_found"
            }
        
        elif status == "no_data_found":
            return {
                "action": "pivot",
                "pivot_focus": self._generate_alternative_focus(question),
                "reason": "no_data_try_different_approach"
            }
        
        elif status == "comprehensive_data":
            # Even with comprehensive data, do one more verification
            return {
                "action": "continue",
                "new_focus": self._generate_verification_focus(question), 
                "reason": "final_verification_check"
            }
        
        else:
            # Default: continue investigating
            return {
                "action": "continue",
                "new_focus": self._generate_contextual_focus(question, iteration),
                "reason": "continue_investigation"
            }
    
    def _generate_deeper_focus(self, question: str) -> str:
        """Generate focus for deeper investigation based on initial findings."""
        if self.findings_history and len(self.findings_history[0]["data"]) > 0:
            first_result = self.findings_history[0]["data"][0]
            if isinstance(first_result, dict):
                # Look at the data structure to decide what to explore
                keys = list(first_result.keys())
                if 'country' in keys:
                    return "historical trends by country"
                elif 'name' in keys:
                    return "detailed performance analysis"
                elif 'count' in str(keys).lower():
                    return "distribution and patterns in the data"
                else:
                    return "relationships and context"
        return "detailed breakdown and context"
    
    def _generate_validation_focus(self, question: str) -> str:
        """Generate focus for validating patterns found."""
        return "cross-reference and validate findings"
    
    def _generate_alternative_focus(self, question: str) -> str:
        """Generate alternative focus when no data found."""
        if "top" in question.lower() or "most" in question.lower():
            return "alternative ranking approach"
        elif "count" in question.lower():
            return "basic counting and aggregation"
        else:
            return "fundamental data exploration"
    
    def _generate_verification_focus(self, question: str) -> str:
        """Generate focus for final verification."""
        return "verify and cross-check results"
    
    def _generate_contextual_focus(self, question: str, iteration: int) -> str:
        """Generate contextual focus based on iteration and question."""
        focuses = [
            "statistical summary",
            "historical context", 
            "comparative analysis",
            "trend validation",
            "final verification"
        ]
        return focuses[min(iteration, len(focuses) - 1)]
    
    def _extract_tables_from_schema(self, schema: str) -> List[str]:
        """Extract table names from schema description."""
        import re
        
        # Pattern to match table definitions in schema
        # Handles formats like: "- tablename(" or "tablename(" or "table: tablename"
        table_patterns = [
            r'- ([a-zA-Z_][a-zA-Z0-9_]*)\(',  # - table_name(
            r'^([a-zA-Z_][a-zA-Z0-9_]*)\(',   # table_name(
            r'table[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)',  # table: table_name
            r'CREATE TABLE ["`]?([a-zA-Z_][a-zA-Z0-9_]*)["`]?',  # CREATE TABLE table_name
        ]
        
        tables = set()
        for pattern in table_patterns:
            matches = re.findall(pattern, schema, re.IGNORECASE | re.MULTILINE)
            tables.update(matches)
        
        # Remove common SQL keywords that might be caught
        sql_keywords = {'SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'UNION', 'JOIN'}
        tables = [t for t in tables if t.upper() not in sql_keywords]
        
        return list(tables)[:10]  # Limit to first 10 tables
    
    def _extract_likely_columns_from_schema(self, schema: str, table_name: str) -> List[str]:
        """Extract likely column names for a specific table from schema."""
        import re
        
        # Find the table definition in schema
        table_pattern = rf'- {re.escape(table_name)}\((.*?)\)'
        match = re.search(table_pattern, schema, re.DOTALL)
        
        if not match:
            # Try alternative patterns
            alt_patterns = [
                rf'{re.escape(table_name)}\((.*?)\)',
                rf'CREATE TABLE ["`]?{re.escape(table_name)}["`]?\s*\((.*?)\)'
            ]
            for pattern in alt_patterns:
                match = re.search(pattern, schema, re.DOTALL | re.IGNORECASE)
                if match:
                    break
        
        if not match:
            return []
        
        columns_text = match.group(1)
        
        # Extract column names (handle various formats)
        column_patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:INTEGER|TEXT|REAL|BLOB|NULL)',  # column_name TYPE
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*,',  # column_name,
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s+[A-Z]',  # column_name TYPE
        ]
        
        columns = set()
        for pattern in column_patterns:
            matches = re.findall(pattern, columns_text, re.IGNORECASE)
            columns.update(matches)
        
        # Common useful column types to look for
        common_columns = []
        columns_list = list(columns)
        
        # Prioritize common patterns
        priority_patterns = ['id', 'name', 'title', 'country', 'date', 'year', 'count', 'total', 'amount']
        for pattern in priority_patterns:
            matching_cols = [col for col in columns_list if pattern.lower() in col.lower()]
            common_columns.extend(matching_cols)
        
        # Add remaining columns
        remaining = [col for col in columns_list if col not in common_columns]
        common_columns.extend(remaining)
        
        return common_columns[:8]  # Limit to 8 most relevant columns
    
    def _generate_schema_aware_fallback(self, question: str, focus: str, iteration: int, schema: str) -> str:
        """Generate a completely database-agnostic fallback query."""
        # Extract available tables from schema
        tables = self._extract_tables_from_schema(schema)
        
        if not tables:
            return "SELECT 1 as no_tables_found"  # Fallback for schema parsing failure
        
        # Pick most relevant table based on question content
        primary_table = self._select_relevant_table(question, tables)
        
        # Extract columns for this table
        columns = self._extract_likely_columns_from_schema(schema, primary_table)
        
        print(f"[AGENTIC] Schema-aware fallback: table={primary_table}, columns={columns[:3]}")
        
        # Generate appropriate query based on question type and iteration
        if ("top" in question.lower() or "most" in question.lower()) and iteration == 0:
            # First iteration: explore the data structure
            return f"SELECT * FROM {primary_table} LIMIT 5"
        
        elif ("top" in question.lower() or "most" in question.lower()) and len(columns) > 1:
            # Ranking query: try to find a groupable column and countable metric
            group_col = self._find_groupable_column(columns)
            if group_col:
                return f"SELECT {group_col}, COUNT(*) as count FROM {primary_table} GROUP BY {group_col} ORDER BY count DESC LIMIT 10"
            else:
                return f"SELECT {columns[0]}, COUNT(*) as count FROM {primary_table} GROUP BY {columns[0]} ORDER BY count DESC LIMIT 10"
        
        elif "count" in question.lower():
            return f"SELECT COUNT(*) as total_count FROM {primary_table}"
        
        else:
            # Generic exploration based on iteration
            if iteration == 0:
                return f"SELECT * FROM {primary_table} LIMIT 5"
            elif iteration == 1:
                return f"SELECT COUNT(*) as record_count FROM {primary_table}"
            elif len(columns) > 0:
                # Try to get distinct values from first meaningful column
                col = self._find_groupable_column(columns) or columns[0]
                return f"SELECT DISTINCT {col} FROM {primary_table} LIMIT 20"
            else:
                return f"SELECT * FROM {primary_table} LIMIT 3"
    
    def _select_relevant_table(self, question: str, tables: List[str]) -> str:
        """Select the most relevant table based on question content."""
        question_lower = question.lower()
        
        # Look for table names mentioned in the question
        for table in tables:
            if table.lower() in question_lower:
                return table
        
        # Look for common entity patterns
        entity_patterns = {
            'country': ['location', 'place', 'circuit', 'venue', 'geography'],
            'user': ['customer', 'person', 'people', 'individual'],
            'product': ['item', 'good', 'merchandise'],
            'order': ['purchase', 'transaction', 'sale'],
            'event': ['race', 'game', 'match', 'contest'],
            'performance': ['result', 'score', 'time', 'speed']
        }
        
        for table in tables:
            table_lower = table.lower()
            for entity_type, keywords in entity_patterns.items():
                if entity_type in table_lower or any(keyword in table_lower for keyword in keywords):
                    return table
        
        # Default to first table
        return tables[0]
    
    def _find_groupable_column(self, columns: List[str]) -> Optional[str]:
        """Find a column suitable for grouping (categorical data)."""
        # Prioritize common groupable column patterns
        groupable_patterns = ['country', 'name', 'type', 'category', 'status', 'region', 'city', 'state']
        
        for pattern in groupable_patterns:
            for col in columns:
                if pattern.lower() in col.lower():
                    return col
        
        # Look for columns that end with common categorical suffixes
        categorical_suffixes = ['_type', '_status', '_category', '_name', '_code']
        for col in columns:
            if any(col.lower().endswith(suffix) for suffix in categorical_suffixes):
                return col
        
        # Avoid numeric/id columns for grouping
        avoid_patterns = ['id', 'count', 'total', 'amount', 'price', 'cost', 'number', 'quantity']
        for col in columns:
            if not any(pattern in col.lower() for pattern in avoid_patterns):
                return col
        
        return None
    
    def _synthesize_final_answer(self, question: str, hypothesis: Dict) -> str:
        """Synthesize final answer from all findings."""
        if not self.findings_history:
            return "❌ No data found to answer your question."
        
        # Create summary of findings
        findings_summary = ""
        for i, finding in enumerate(self.findings_history):
            data_preview = ""
            if finding["data"] and len(finding["data"]) > 0:
                first_row = finding["data"][0]
                if isinstance(first_row, dict):
                    # Show key-value pairs from first row
                    preview_items = list(first_row.items())[:3]
                    data_preview = ", ".join([f"{k}: {v}" for k, v in preview_items])
            
            findings_summary += f"Finding {i+1}: {len(finding['data'])} rows ({data_preview})\n"
        
        prompt = f"""
Synthesize a final answer from this agentic investigation.

Question: "{question}"
Investigation Findings:
{findings_summary}

Provide a concise answer (max 1000 characters) in this format:

🤖 *Question Answer*

📊 *Key Finding:*
[Direct answer with specific data]

💡 *Details:*
[Supporting information from the investigation]

*Investigated through {len(self.findings_history)} targeted queries*

Guidelines:
- Answer the specific question asked
- Use actual data from findings
- Be confident and direct
- Include specific numbers/names where available
"""
        
        response = self._call_llm(prompt)
        return response if response else "❌ Could not synthesize final answer."
    
    def _execute_safe_query(self, query: str) -> Optional[List[Dict]]:
        """Execute query with same safety as regular system."""
        try:
            from core.guardrails import sanitize_sql
            is_safe, safe_query, reason = sanitize_sql(query, 50)  # Limit to 50 rows for agentic
            
            if not is_safe:
                print(f"[AGENTIC] Query failed guardrails: {reason}")
                return None
            
            result = DatabaseService.execute_query(safe_query)
            if isinstance(result, dict) and 'data' in result:
                return result['data']
            else:
                print(f"[AGENTIC] Unexpected result format: {result}")
                return None
                
        except Exception as e:
            print(f"[AGENTIC] Error executing query: {e}")
            return None
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM with current backend."""
        try:
            if config.LLM_BACKEND == "groq":
                return LLMService._call_groq(prompt)
            else:
                return LLMService._call_ollama(prompt)
        except Exception as e:
            print(f"[AGENTIC] LLM call failed: {e}")
            return None
    
    def _call_llm_for_json(self, prompt: str, step_name: str) -> Optional[Dict]:
        """Call LLM and parse JSON response."""
        try:
            response = self._call_llm(prompt)
            if not response:
                return None
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            print(f"[AGENTIC] No valid JSON in {step_name} response: {response[:200]}...")
            return None
            
        except json.JSONDecodeError as e:
            print(f"[AGENTIC] JSON parsing failed for {step_name}: {e}")
            return None
        except Exception as e:
            print(f"[AGENTIC] Error in {step_name}: {e}")
            return None


# Global instance for use in routes
true_agentic_analyst = TrueAgenticAnalyst()