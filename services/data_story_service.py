"""
Data Story Service for CircularQuery.

Generates comprehensive data stories with AI-generated visualizations.
Combines narrative analysis with whiteboard-style visual presentations.
"""
import os
import time
import hashlib
import torch
from typing import Dict, List, Any, Optional, Tuple
from config.settings import config
from services.database import DatabaseService
from services.llm import LLMService


class DataStoryService:
    """Service for creating comprehensive data stories with AI-generated visuals."""
    
    def __init__(self):
        """Initialize the data story service."""
        self.pipeline = None
        self._pipeline_loaded = False
        
    def _load_image_pipeline(self):
        """Lazy load the image generation pipeline."""
        if self._pipeline_loaded:
            return
            
        try:
            # Try multiple approaches to load the image generation pipeline
            print("[DATASTORY] Loading Z-Image-Turbo pipeline...")
            
            # First try the specific Z-Image pipeline
            try:
                from diffusers import ZImagePipeline
                
                self.pipeline = ZImagePipeline.from_pretrained(
                    "Tongyi-MAI/Z-Image-Turbo",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,  # Match original exactly
                )
                print("[DATASTORY] Loaded Z-Image-Turbo with AutoPipeline")
            except Exception as e1:
                print(f"[DATASTORY] AutoPipeline failed ({e1}), trying DiffusionPipeline...")
                # Fallback to generic diffusion pipeline
                from diffusers import DiffusionPipeline
                self.pipeline = DiffusionPipeline.from_pretrained(
                    "Tongyi-MAI/Z-Image-Turbo",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
                print("[DATASTORY] Loaded Z-Image-Turbo with DiffusionPipeline")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
                print(f"[DATASTORY] Pipeline loaded on CUDA device")
            else:
                print("[DATASTORY] CUDA not available, using CPU (will be slower)")
            
            # Enable memory efficient attention
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("[DATASTORY] Enabled xformers memory optimization")
                except:
                    pass
            
            self._pipeline_loaded = True
            print("[DATASTORY] Pipeline ready for image generation!")
            
        except Exception as e:
            print(f"[DATASTORY] Failed to load image pipeline: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None
            # Set as loaded to prevent repeated attempts
            self._pipeline_loaded = True
            
    def create_data_story(self, question: str, schema: str) -> Dict[str, Any]:
        """
        Create a comprehensive data story from a business question.
        
        Args:
            question: Business question to investigate
            schema: Database schema for context
            
        Returns:
            Dict containing story text, visual prompt, and image path
        """
        print(f"[DATASTORY] Creating data story for: {question}")
        
        try:
            # Step 1: Generate SQL and get data
            sql_query = LLMService.get_sql_query(question, schema)
            if not sql_query:
                return {"error": "Could not generate SQL query for this question"}
                
            result = DatabaseService.execute_query(sql_query)
            if "error" in result:
                return {"error": f"SQL error: {result['error']}"}
                
            data = result.get("data", [])
            if not data:
                return {"error": "No data found for this question"}
            
            print(f"[DATASTORY] Found {len(data)} rows of data")
            
            # Step 2: Generate comprehensive analytical report
            print("[DATASTORY] Generating analytical report...")
            analytical_report = self._generate_analytical_report(question, sql_query, data, schema)
            
            # If analytical report fails, create a basic one from the data
            if not analytical_report or len(analytical_report) < 50:
                print("[DATASTORY] Fallback: Creating basic analytical summary...")
                analytical_report = self._create_basic_analysis(question, data)
            
            # Step 3: Create structured data story
            story_structure = self._create_story_structure(question, data, analytical_report)
            
            # Step 4: Generate visual prompt for AI image generation
            visual_prompt = self._generate_visual_prompt(question, data, story_structure)
            
            # Step 5: Generate AI visualization
            image_path = self._generate_visualization(visual_prompt, question, story_structure)
            
            return {
                "story_text": story_structure,
                "analytical_report": analytical_report,
                "visual_prompt": visual_prompt,
                "image_path": image_path,
                "sql_query": sql_query,
                "data_rows": len(data)
            }
            
        except Exception as e:
            print(f"[DATASTORY] Error creating data story: {e}")
            return {"error": f"Failed to create data story: {str(e)}"}
    
    def _generate_analytical_report(self, question: str, sql_query: str, data: List[Dict], schema: str) -> str:
        """Generate comprehensive analytical report using data analyst agent."""
        
        # Load the data analyst agent prompt
        analyst_prompt_path = "agents/data_analyst_agent.txt"
        try:
            with open(analyst_prompt_path, 'r') as f:
                analyst_prompt_template = f.read()
        except FileNotFoundError:
            analyst_prompt_template = """Generate a comprehensive analytical report of approximately 1200 words for this data analysis."""
        
        # Prepare data summary for analysis
        data_summary = self._create_data_summary(data)
        
        # Create analysis prompt
        analysis_prompt = f"""
{analyst_prompt_template}

ANALYSIS REQUEST:
Question: {question}
SQL Query: {sql_query}
Data Summary: {data_summary}
Schema Context: {schema[:1000]}...

Please provide a comprehensive analytical report following the structured format outlined in the prompt.
Focus on extracting insights, patterns, and actionable recommendations from this data.
"""
        
        print("[DATASTORY] Generating analytical report...")
        report = LLMService.generate_insights(question, sql_query, data, schema, custom_prompt=analysis_prompt)
        
        return report or "Analysis could not be generated at this time."
    
    def _create_story_structure(self, question: str, data: List[Dict], analytical_report: str) -> str:
        """Create structured 5-part data story using story teller agent."""
        
        # Extract key data points for natural storytelling
        key_insights = self._extract_detailed_insights(data)
        
        # Create focused story prompt that emphasizes natural flow
        story_prompt = f"""
You are a data storyteller creating a compelling business narrative. Transform this analysis into a natural, engaging story that flows smoothly without rigid section headers.

ANALYSIS TO TRANSFORM:
Question: {question}
Key Findings: {key_insights}
Context: {analytical_report[:800]}...

STORYTELLING REQUIREMENTS:
- Write as a natural, flowing narrative (not a structured report)
- Start with context and build to insights organically  
- Include specific numbers and statistics naturally in the text
- Focus on the most surprising or important patterns
- End with clear implications for decision-making
- Keep it concise and punchy (250-350 words max for Slack readability)
- NO section headers like "Part 1", "Part 2" etc - just smooth narrative flow
- Use shorter paragraphs for better mobile/Slack formatting

Write a compelling data story that someone would want to read and share with colleagues:
"""
        
        print("[DATASTORY] Creating natural story narrative...")
        story_structure = LLMService.generate_text_response(story_prompt)
        
        return story_structure or self._create_default_natural_story(question, key_insights)
    
    def _generate_visual_prompt(self, question: str, data: List[Dict], story_structure: str) -> str:
        """Generate whiteboard-style visual prompt using LLM meta-prompt approach."""
        
        # Step 1: Extract data into JSON format  
        data_extraction_prompt = f"""
Extract the key data insights from this analysis into clean JSON format for visualization:

QUESTION: {question}
DATA: {data}
STORY: {story_structure[:500]}...

Create a JSON object with these exact fields:
{{
  "title": "Short compelling title",
  "winner": "Name of top performer",
  "winner_value": "Numerical value",
  "metric_name": "What is being measured",
  "top_3": [
    {{"name": "First place name", "value": "number"}},
    {{"name": "Second place name", "value": "number"}}, 
    {{"name": "Third place name", "value": "number"}}
  ],
  "key_insight": "One surprising finding in 8 words or less",
  "total_analyzed": "Number of items analyzed"
}}

Extract ONLY the actual data from the analysis. Return valid JSON only:
"""
        
        print("[DATASTORY] Extracting structured data for visualization...")
        json_response = LLMService.generate_text_response(data_extraction_prompt)
        
        # Parse JSON and create data summary
        try:
            import json
            data_json = json.loads(json_response) if json_response else {}
            print(f"[DATASTORY] Extracted data JSON: {data_json}")
        except Exception as e:
            print(f"[DATASTORY] JSON parsing failed, using fallback data: {e}")
            data_json = self._extract_fallback_json(data, question.title())
        
        # Step 2: Format data for the meta-prompt
        winner = data_json.get('winner', 'Top Performer')
        winner_value = data_json.get('winner_value', 'N/A')
        metric = data_json.get('metric_name', 'Performance')
        top_3 = data_json.get('top_3', [])
        title = data_json.get('title', question.title())
        insight = data_json.get('key_insight', 'Analysis complete')
        
        data_summary = f"""
Title: {title}
Winner: {winner} ({winner_value} {metric})
Rankings: {', '.join([f"{item.get('name', 'N/A')}: {item.get('value', 'N/A')}" for item in top_3[:3]])}
Key Insight: {insight}
Total Analyzed: {data_json.get('total_analyzed', len(data))} items
"""
        
        # Step 3: Use LLM to generate the whiteboard visual prompt
        meta_prompt = f"""
Using the data below, generate a short text-to-image prompt that describes a hand-drawn whiteboard-style visual showing the key values, comparisons, and insights. Include simple bars or arrows, brief handwritten notes, and 2-3 key takeaways. Keep it clean, minimal, and business-whiteboard in style. End with: white background, marker sketch.

Data:
{data_summary.strip()}
"""
        
        print("[DATASTORY] Generating whiteboard visual prompt from data...")
        generated_prompt = LLMService.generate_text_response(meta_prompt)
        
        if generated_prompt and len(generated_prompt) > 50:
            visual_prompt = generated_prompt.strip()
            print(f"[DATASTORY] Generated visual prompt: {visual_prompt[:150]}...")
        else:
            # Fallback prompt if LLM generation fails
            visual_prompt = f"""
Hand-drawn whiteboard showing "{title}". Simple bar chart with {winner} at {winner_value} {metric} (longest bar), followed by smaller bars for runners-up. Arrow pointing to winner with "TOP!" note. Key stats in corners: "{insight}". Clean business whiteboard style, handwritten labels, simple drawings. White background, marker sketch.
"""
            print("[DATASTORY] Using fallback visual prompt")
        
        return visual_prompt
    
    def _generate_visualization(self, visual_prompt: str, question: str, story_text: str = "") -> Optional[str]:
        """Generate AI visualization using the image pipeline."""
        
        try:
            # Load pipeline if needed
            self._load_image_pipeline()
            
            if not self.pipeline:
                print("[DATASTORY] Image pipeline not available, skipping visualization")
                return None
            
            print("[DATASTORY] Generating AI visualization...")
            print(f"[DATASTORY] Visual prompt length: {len(visual_prompt)} characters")
            print(f"[DATASTORY] Using question: {question}")
            
            # Extract title from visual prompt for logging
            title_start = visual_prompt.find("titled '") + 8
            title_end = visual_prompt.find("'", title_start)
            if title_start > 7 and title_end > title_start:
                extracted_title = visual_prompt[title_start:title_end]
                print(f"[DATASTORY] Infographic title: {extracted_title}")
            else:
                print(f"[DATASTORY] Using fallback title: {question}")
            
            # Generate image with error handling
            with torch.no_grad():
                # Try different inference parameters based on model capabilities
                try:
                    # First try with Z-Image-Turbo specific parameters
                    result = self.pipeline(
                        prompt=visual_prompt,
                        height=1024,
                        width=1024,
                        num_inference_steps=9,  # Z-Image-Turbo is optimized for few steps
                        guidance_scale=0.0,     # Z-Image-Turbo often works well with no guidance
                        generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42),
                    )
                    
                    if hasattr(result, 'images'):
                        image = result.images[0]
                    else:
                        image = result
                        
                    print("[DATASTORY] Image generated successfully with Z-Image parameters")
                    
                except Exception as gen_error:
                    print(f"[DATASTORY] Z-Image parameters failed ({gen_error}), trying standard parameters...")
                    # Fallback to more standard parameters
                    result = self.pipeline(
                        prompt=visual_prompt,
                        height=768,  # Smaller size for compatibility
                        width=768,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42),
                    )
                    
                    if hasattr(result, 'images'):
                        image = result.images[0]
                    else:
                        image = result
                        
                    print("[DATASTORY] Image generated successfully with standard parameters")
            
            # Ensure exports directory exists
            exports_dir = getattr(config, 'EXPORTS_DIR', 'exports')
            os.makedirs(exports_dir, exist_ok=True)
            
            # Save with hashed filename
            timestamp = int(time.time())
            question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
            filename = f"datastory_{timestamp}_{question_hash}.png"
            filepath = os.path.join(exports_dir, filename)
            
            image.save(filepath)
            print(f"[DATASTORY] Visualization saved: {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"[DATASTORY] Error generating visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_data_summary(self, data: List[Dict]) -> str:
        """Create a summary of the data for analysis."""
        if not data:
            return "No data available"
            
        summary_parts = []
        summary_parts.append(f"Dataset contains {len(data)} rows")
        
        if data:
            first_row = data[0]
            summary_parts.append(f"Columns: {list(first_row.keys())}")
            
            # Sample a few values
            if len(data) >= 3:
                summary_parts.append("Sample data:")
                for i, row in enumerate(data[:3]):
                    summary_parts.append(f"  Row {i+1}: {dict(list(row.items())[:3])}")
        
        return "\n".join(summary_parts)
    
    def _extract_detailed_insights(self, data: List[Dict]) -> str:
        """Extract detailed insights from the data for natural story creation."""
        if not data:
            return "No data insights available"
            
        insights = []
        insights.append(f"Dataset contains {len(data)} records")
        
        if data:
            first_row = data[0]
            columns = list(first_row.keys())
            
            # Identify top performer and key metrics
            if len(data) >= 1:
                top_item = first_row
                insights.append(f"Top entry: {list(top_item.values())[0] if top_item.values() else 'N/A'}")
                
            # Look for numerical columns for statistics
            numerical_columns = []
            for col in columns:
                if any(isinstance(row.get(col), (int, float)) for row in data[:3]):
                    numerical_columns.append(col)
            
            if numerical_columns:
                insights.append(f"Key metrics tracked: {', '.join(numerical_columns[:3])}")
                
            # Sample a few key values for context
            if len(data) >= 3:
                top_3_names = [str(list(row.values())[0]) for row in data[:3]]
                insights.append(f"Top 3 entries: {', '.join(top_3_names)}")
        
        return " | ".join(insights)
    
    def _extract_infographic_stats(self, data: List[Dict], story_text: str) -> str:
        """Extract specific statistics for infographic generation."""
        if not data:
            return "No data available for visualization"
            
        stats = []
        
        # Basic stats
        stats.append(f"TOTAL RECORDS: {len(data)}")
        
        if data:
            first_row = data[0]
            columns = list(first_row.keys())
            
            # Identify value column (likely the numerical one)
            value_col = None
            for col in columns[1:]:  # Skip first column (usually name)
                if isinstance(first_row.get(col), (int, float)):
                    value_col = col
                    break
            
            # Enhanced top performer details
            if len(first_row) >= 2:
                name = str(list(first_row.values())[0])
                value = list(first_row.values())[1]
                stats.append(f"🏆 WINNER: {name}")
                if isinstance(value, (int, float)):
                    stats.append(f"💯 WINNING SCORE: {value:,}")
                    if value_col:
                        stats.append(f"📊 METRIC: {value_col.replace('_', ' ').title()}")
            
            # Detailed competition breakdown
            if len(data) >= 2 and value_col:
                # Top 3 with performance details
                stats.append("🏁 LEADERBOARD:")
                for i, row in enumerate(data[:3]):
                    name = str(list(row.values())[0])
                    value = row.get(value_col, 0)
                    medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                    stats.append(f"{medal} {name}: {value:,}")
                
                # Competition gaps
                if len(data) >= 2:
                    first_val = data[0].get(value_col, 0)
                    second_val = data[1].get(value_col, 0)
                    if isinstance(first_val, (int, float)) and isinstance(second_val, (int, float)):
                        gap = first_val - second_val
                        gap_pct = (gap / second_val * 100) if second_val > 0 else 0
                        stats.append(f"📈 LEAD MARGIN: {gap:,} points ({gap_pct:.1f}%)")
                
                # Total performance
                total = sum(row.get(value_col, 0) for row in data if isinstance(row.get(value_col), (int, float)))
                if total > 0:
                    stats.append(f"📊 COMBINED TOTAL: {total:,}")
        
        # Extract key insights from story text for visual elements
        if story_text and len(story_text) > 100:
            # Look for specific patterns in the story
            import re
            percentages = re.findall(r'\d+(?:\.\d+)?%', story_text)
            if percentages:
                stats.append(f"📈 KEY PERCENTAGES: {', '.join(percentages[:3])}")
            
            # Look for comparative language
            if "dominance" in story_text.lower() or "dominant" in story_text.lower():
                stats.append("🚀 TREND: Dominant Performance")
            elif "close" in story_text.lower() or "tight" in story_text.lower():
                stats.append("⚖️ TREND: Close Competition")
        
        return "\n".join(stats)
    
    def _create_basic_analysis(self, question: str, data: List[Dict]) -> str:
        """Create basic analysis when LLM analytical report fails."""
        if not data:
            return f"Analysis of {question} found no data to examine."
        
        analysis_parts = []
        analysis_parts.append(f"Analysis of {question} reveals {len(data)} key data points.")
        
        # Extract top performers and key metrics
        if data:
            first_row = data[0]
            columns = list(first_row.keys())
            
            # Identify name and value columns
            name_col = columns[0] if columns else "item"
            value_cols = [col for col in columns[1:] if isinstance(first_row.get(col), (int, float))]
            
            if value_cols:
                value_col = value_cols[0]
                top_value = first_row.get(value_col, 0)
                top_name = first_row.get(name_col, "Unknown")
                
                analysis_parts.append(f"The clear leader is {top_name} with {top_value:,} {value_col.replace('_', ' ')}.")
                
                # Compare with others if available
                if len(data) > 1:
                    second_value = data[1].get(value_col, 0)
                    gap = top_value - second_value if isinstance(top_value, (int, float)) and isinstance(second_value, (int, float)) else 0
                    if gap > 0:
                        analysis_parts.append(f"This represents a significant {gap:,} point advantage over the second-place competitor.")
                
                # Overall summary
                total = sum(row.get(value_col, 0) for row in data if isinstance(row.get(value_col), (int, float)))
                if total > 0:
                    analysis_parts.append(f"Combined, these top performers account for {total:,} total {value_col.replace('_', ' ')}, demonstrating the competitive landscape in this analysis.")
        
        return " ".join(analysis_parts)
    
    def _extract_visual_elements(self, data: List[Dict]) -> str:
        """Extract visual elements from data for prompt generation."""
        if not data:
            return "No data for visualization"
            
        elements = []
        
        # Determine chart type based on data structure
        if len(data) <= 10:
            elements.append("Bar chart showing all data points ranked by value")
        else:
            elements.append("Bar chart showing top 10 items")
            
        # Add specific values if available
        if data:
            first_row = data[0]
            if len(first_row) >= 2:
                values = list(first_row.values())
                elements.append(f"Top value: {values[1]} for {values[0]}")
        
        # Suggest specific visual elements
        elements.append("Callout boxes highlighting key statistics")
        elements.append("Percentage comparisons between top performers")
        elements.append("Trend indicators (arrows, percentages)")
        
        return " | ".join(elements)
    
    def _create_default_natural_story(self, question: str, key_insights: str) -> str:
        """Create a default natural story if LLM generation fails."""
        return f"""
When we examined {question.lower()}, several fascinating patterns emerged from the data. {key_insights}

The most striking discovery reveals how performance varies significantly across different categories, with clear leaders emerging from the pack. What makes this particularly interesting is not just who's at the top, but the gap between first and subsequent positions.

Looking deeper into the numbers, we can see distinct tiers of performance that suggest different strategic approaches or resource allocations. The data shows that consistency often trumps occasional peaks, with steady performers maintaining their positions while others show more volatile patterns.

These insights have important implications for decision-making, suggesting that sustained excellence requires a different approach than sporadic high performance. The patterns we've identified can help inform future strategies and resource allocation decisions.
"""


# Initialize the service
data_story_service = DataStoryService()