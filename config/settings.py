"""
Configuration management for CircularQuery.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class."""
    
    # Slack Configuration
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
    SLACK_API_BASE = "https://slack.com/api"
    
    # LLM Configuration
    LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
    
    # Ollama Configuration (local)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite4")
    
    # Groq Configuration (cloud)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # RAG Configuration
    USE_RAG = os.getenv("USE_RAG", "false").lower() in ("1", "true", "yes")
    RAG_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "./rag_index")
    RAG_TOP_K_TABLES = int(os.getenv("RAG_TOP_K_TABLES", "5"))
    RAG_TOP_N_COLUMNS = int(os.getenv("RAG_TOP_N_COLUMNS", "8"))

    # Agent pacing (to mitigate rate limits during RAG-enabled investigations)
    AGENT_MAX_ITER_RAG = int(os.getenv("AGENT_MAX_ITER_RAG", "3"))
    AGENT_ITERATION_DELAY = float(os.getenv("AGENT_ITERATION_DELAY", "1.0"))

    # Routing /askdb behavior
    ASKDB_FORCE_AGENTIC = os.getenv("ASKDB_FORCE_AGENTIC", "false").lower() in ("1", "true", "yes")
    
    # Database Configuration
    SQLITE_PATH = os.getenv("SQLITE_PATH", "./chinook.db")
    SCHEMA_YAML_PATH = os.getenv("SCHEMA_YAML_PATH", "./schema.yaml")
    
    # Application Configuration
    EXPORTS_DIR = "./exports"
    ROWS_PER_PAGE = 12
    # Allow overriding default query limit via environment variable
    # Set to 0 or negative value to disable limit entirely
    DEFAULT_LIMIT = int(os.getenv("QUERY_LIMIT", "500"))
    LLM_TIMEOUT = 120
    
    # Optional Configuration
    NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "")
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.SLACK_BOT_TOKEN:
            raise ValueError("SLACK_BOT_TOKEN is required")
        
        if not os.path.exists(cls.SQLITE_PATH):
            raise FileNotFoundError(f"SQLite database not found at {cls.SQLITE_PATH}")
        
        # Validate LLM backend configuration
        if cls.LLM_BACKEND not in ["ollama", "groq"]:
            raise ValueError(f"LLM_BACKEND must be 'ollama' or 'groq', got '{cls.LLM_BACKEND}'")
        
        if cls.LLM_BACKEND == "groq" and not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required when LLM_BACKEND is 'groq'")
        
        # Create exports directory if it doesn't exist
        os.makedirs(cls.EXPORTS_DIR, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

# Default configuration
config = DevelopmentConfig()