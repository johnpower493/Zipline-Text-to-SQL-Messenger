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
        self.max_iterations = 5
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
            
            for iteration in range(self.max_iterations):
                print(f"[AGENTIC] --- Iteration {iteration + 1}/{self.max_iterations} ---")
                
                # Generate next query based on current understanding
                next_query = self._generate_focused_query(
                    current_focus, question, schema, iteration
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
    
    def _generate_focused_query(self, focus: str, question: str, schema: str, iteration: int) -> Optional[str]:
        """Generate a focused query for current investigation step."""
        context = ""
        if self.findings_history:
            latest = self.findings_history[-1]
            context = f"Previous finding: {len(latest['data'])} rows from {latest['focus']}"
        
        prompt = f"""
Generate a simple SQLite query for this investigation.

Question: "{question}"
Focus: "{focus}"
Schema preview: {schema[:600]}

Examples for similar questions:
- Team winner: SELECT c.name, SUM(cr.points) as points FROM constructors c JOIN constructor_results cr ON c.constructorId = cr.constructorId JOIN races r ON cr.raceId = r.raceId WHERE r.year = 2020 GROUP BY c.constructorId ORDER BY points DESC LIMIT 5
- Driver winner: SELECT d.forename, d.surname, SUM(r.points) as points FROM drivers d JOIN results r ON d.driverId = r.driverId JOIN races ra ON r.raceId = ra.raceId WHERE ra.year = 2020 GROUP BY d.driverId ORDER BY points DESC LIMIT 5
- Overview: SELECT COUNT(*) FROM constructors

Generate ONE simple query. Respond with ONLY the SQL:
"""
        
        response = self._call_llm(prompt)
        
        if response and response.strip().upper().startswith("SELECT"):
            query = response.strip().rstrip(";")
            print(f"[AGENTIC] Generated query: {query}")
            return query
        
        print(f"[AGENTIC] Invalid query response: {response}")
        
        # Intelligent fallback based on question content
        if "team" in question.lower() and ("won" in question.lower() or "2020" in question.lower()):
            fallback = "SELECT c.name, SUM(cr.points) as total_points FROM constructors c JOIN constructor_results cr ON c.constructorId = cr.constructorId JOIN races r ON cr.raceId = r.raceId WHERE r.year = 2020 GROUP BY c.constructorId ORDER BY total_points DESC LIMIT 5"
            print(f"[AGENTIC] Using team winner fallback")
            return fallback
        elif "driver" in question.lower() and ("won" in question.lower() or "champion" in question.lower()):
            year = "2021" if "2021" in question else "2020"
            fallback = f"SELECT d.forename, d.surname, COUNT(*) as races FROM drivers d JOIN results r ON d.driverId = r.driverId GROUP BY d.driverId ORDER BY races DESC LIMIT 5"
            print(f"[AGENTIC] Using driver fallback")
            return fallback
        else:
            fallback = "SELECT COUNT(*) as record_count FROM races WHERE year = 2020"
            print(f"[AGENTIC] Using basic fallback")
            return fallback
    
    def _analyze_current_findings(self, question: str, hypothesis: Dict) -> Dict:
        """Analyze current findings to understand what we've learned."""
        if not self.findings_history:
            return {"status": "no_data", "confidence": 0}
        
        latest_finding = self.findings_history[-1]
        total_findings = len(self.findings_history)
        
        # Simple analysis based on data availability and question type
        if len(latest_finding["data"]) == 0:
            return {
                "status": "no_data_found",
                "confidence": 0,
                "insight": "No data returned from last query"
            }
        elif "won" in question.lower() or "most" in question.lower() or "top" in question.lower():
            # Looking for rankings/winners
            return {
                "status": "ranking_found",
                "confidence": 80,
                "insight": f"Found {len(latest_finding['data'])} results for ranking analysis"
            }
        elif total_findings >= 2:
            return {
                "status": "sufficient_data",
                "confidence": 90,
                "insight": f"Have {total_findings} findings with good data coverage"
            }
        else:
            return {
                "status": "need_more_data",
                "confidence": 50,
                "insight": "Need additional data to form complete answer"
            }
    
    def _decide_next_action(self, analysis: Dict, question: str, iteration: int) -> Dict:
        """Decide what to do next based on current analysis."""
        status = analysis.get("status", "unknown")
        confidence = analysis.get("confidence", 0)
        
        if iteration >= self.max_iterations - 1:
            return {"action": "complete", "reason": "max_iterations_reached"}
        
        if confidence >= 80 and "ranking_found" in status:
            return {"action": "complete", "reason": "sufficient_confidence"}
        
        if status == "no_data_found":
            # Try different approach
            if "driver" in question.lower():
                return {
                    "action": "pivot", 
                    "pivot_focus": "driver race participation",
                    "reason": "pivot_to_driver_data"
                }
            elif "team" in question.lower() or "constructor" in question.lower():
                return {
                    "action": "pivot",
                    "pivot_focus": "constructor performance",
                    "reason": "pivot_to_team_data"
                }
            else:
                return {
                    "action": "pivot",
                    "pivot_focus": "basic dataset overview",
                    "reason": "pivot_to_overview"
                }
        
        if status == "need_more_data":
            # Continue investigation with refined focus
            if "driver" in question.lower():
                return {
                    "action": "continue",
                    "new_focus": "driver performance details",
                    "reason": "need_driver_details"
                }
            else:
                return {
                    "action": "continue", 
                    "new_focus": "detailed analysis",
                    "reason": "need_more_context"
                }
        
        return {"action": "complete", "reason": "default_completion"}
    
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