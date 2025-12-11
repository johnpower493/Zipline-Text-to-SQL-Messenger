"""
Enhanced SQL guardrails for CircularQuery.
Provides comprehensive security validation for SQL queries.
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

ALLOWED_SELECT = re.compile(r"^\s*(WITH\s+.+?AS\s*\(.*?\)\s*)*SELECT\b", re.IGNORECASE | re.DOTALL)


def sanitize_sql(sql: str, default_limit: int = 500) -> Tuple[bool, str, str]:
    """
    Sanitize SQL query for safety.
    
    Returns (ok, sql_out, reason_if_not_ok).
    - Only allow SELECT (and optional CTE `WITH`…).
    - Strip trailing semicolon.
    - If no LIMIT present, append LIMIT <default_limit>.
    - Enhanced security checks.
    
    Args:
        sql: Input SQL string
        default_limit: Default LIMIT to apply if none present
        
    Returns:
        Tuple of (is_safe, sanitized_sql, rejection_reason)
    """
    if not sql or not isinstance(sql, str):
        logger.warning("Empty or non-string SQL provided")
        return False, "", "empty_sql"

    # Clean up common formatting issues
    s = sql.strip().strip("`")
    s = re.sub(r"^sql\s*\n", "", s, flags=re.IGNORECASE).strip()
    s = s.rstrip(";").strip()
    
    # Check for empty query after cleanup
    if not s:
        return False, "", "empty_sql_after_cleanup"

    # Must start with SELECT (optionally WITH…)
    if not ALLOWED_SELECT.match(s):
        logger.warning(f"Non-SELECT query rejected: {s[:100]}...")
        return False, "", "non_select_or_unsafe"

    # Enhanced forbidden keywords check
    forbidden_patterns = [
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|ATTACH|DETACH)\b",
        r"\b(PRAGMA|VACUUM|REINDEX|ANALYZE)\b",
        r"\b(CREATE|REPLACE)\b",
        r"\b(EXEC|EXECUTE)\b",
        r"--",  # SQL comments
        r"/\*",  # Block comments
    ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, s, flags=re.IGNORECASE):
            logger.warning(f"Forbidden pattern found: {pattern} in query: {s[:100]}...")
            return False, "", "forbidden_keyword"

    # Check for suspicious characters or sequences
    suspicious_patterns = [
        r";.*\w",  # Multiple statements (multi-statement)
        r"\b(pg_|information_schema|sys\.|mysql\.)",  # System tables/schemas
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, s, flags=re.IGNORECASE):
            logger.warning(f"Suspicious pattern found: {pattern} in query: {s[:100]}...")
            return False, "", "suspicious_pattern"

    # Add LIMIT if not present (unless limit is disabled)
    if not re.search(r"\bLIMIT\b\s+\d+", s, flags=re.IGNORECASE):
        if default_limit > 0:  # Only add limit if not disabled
            s = f"{s} LIMIT {default_limit}"
    else:
        # Check if existing limit is reasonable (unless limit checking is disabled)
        if default_limit > 0:  # Only enforce limit checks if not disabled
            limit_match = re.search(r"\bLIMIT\b\s+(\d+)", s, flags=re.IGNORECASE)
            if limit_match and int(limit_match.group(1)) > 10000:
                logger.warning(f"Excessive LIMIT value: {limit_match.group(1)}")
                return False, "", "excessive_limit"

    logger.info(f"SQL query sanitized successfully: {s[:100]}...")
    return True, s, ""


def validate_table_name(table_name: str) -> bool:
    """
    Validate that a table name is safe for use in queries.
    
    Args:
        table_name: Table name to validate
        
    Returns:
        True if table name is safe, False otherwise
    """
    if not table_name or not isinstance(table_name, str):
        return False
        
    # Allow only alphanumeric, underscore, and basic characters
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", table_name):
        return False
        
    # Reject system table patterns
    system_patterns = [
        r"^sqlite_",
        r"^pg_",
        r"^information_schema",
        r"^sys",
        r"^mysql",
    ]
    
    for pattern in system_patterns:
        if re.match(pattern, table_name, re.IGNORECASE):
            return False
            
    return True