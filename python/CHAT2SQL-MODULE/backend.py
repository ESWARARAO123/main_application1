from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import logging
import re
import requests
import os
from psycopg2 import pool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama Configuration - Auto-detect Docker environment
# Use host.docker.internal when running in Docker, localhost otherwise
OLLAMA_HOST = "host.docker.internal" if os.path.exists('/.dockerenv') else "localhost"
OLLAMA_API_URL = f"http://{OLLAMA_HOST}:11434/api/generate"
MODEL_NAME = "mistral"

# Log configuration on startup
logger.info(f"Ollama API URL: {OLLAMA_API_URL}")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Running in Docker: {os.path.exists('/.dockerenv')}")
logger.info(f"Ollama Host: {OLLAMA_HOST}")

# FastAPI App Setup
app = FastAPI(
    title="SQL Executor API",
    description="API for executing SQL queries using Ollama for natural language understanding",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Expose all headers
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise

# Pydantic Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    sql: str
    data: List[Dict[str, Any]]
    columns: List[str]

def get_app_db_connection():
    return psycopg2.connect(
        host=os.environ.get("APP_DB_HOST", "localhost"),
        database=os.environ.get("APP_DB_NAME", "copilot"),
        user=os.environ.get("APP_DB_USER", "postgres"),
        password=os.environ.get("APP_DB_PASSWORD", "root"),
        port=int(os.environ.get("APP_DB_PORT", 5432))
    )

def get_user_db_config(user_id):
    conn = get_app_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT host, database, db_user, db_password, port FROM database_details WHERE user_id = %s", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "host": row[0],
            "database": row[1],
            "user": row[2],
            "password": row[3],
            "port": row[4]
        }
    return None

def save_user_db_config_to_db(user_id: str, config: dict):
    conn = get_app_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO database_details (user_id, host, database, db_user, db_password, port, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id) DO UPDATE SET
            host = EXCLUDED.host,
            database = EXCLUDED.database,
            db_user = EXCLUDED.db_user,
            db_password = EXCLUDED.db_password,
            port = EXCLUDED.port,
            updated_at = NOW()
    """, (user_id, config['host'], config['database'], config['user'], config['password'], config['port']))
    conn.commit()
    conn.close()

def get_user_id_from_username(username: str):
    conn = get_app_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]  # UUID
    return None

# --- API endpoints for DB config ---
@app.get("/api/db-config")
async def get_db_config(x_username: Optional[str] = Header(None)):
    username = x_username or 'default'
    user_id = get_user_id_from_username(username)
    if not user_id:
        return JSONResponse(content={}, headers={"Content-Type": "application/json"})
    config = get_user_db_config(user_id)
    if not config:
        return JSONResponse(content={}, headers={"Content-Type": "application/json"})
    return JSONResponse(content=config, headers={"Content-Type": "application/json"})

@app.post("/api/db-config")
async def set_db_config(request: Request, x_username: Optional[str] = Header(None), x_db_test: Optional[str] = Header(None)):
    username = x_username or 'default'
    user_id = get_user_id_from_username(username)
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "User not found."})
    body = await request.json()
    required = ['host', 'database', 'user', 'password', 'port']
    if not all(k in body and body[k] for k in required):
        logger.error(f"[DB-CONFIG] Missing fields for user {username}")
        return JSONResponse(status_code=400, content={"error": "All fields are required."})
    # Test connection
    try:
        logger.info(f"[DB-CONFIG] Trying to connect with: host={body['host']}, db={body['database']}, user={body['user']}, port={body['port']}")
        test_conn = psycopg2.connect(
            host=body['host'],
            database=body['database'],
            user=body['user'],
            password=body['password'],
            port=body['port'],
            sslmode='disable'
        )
        test_conn.close()
    except Exception as e:
        logger.error(f"[DB-CONFIG] Connection failed for user {username}: {repr(e)}")
        return JSONResponse(status_code=400, content={"error": f"Connection failed: {str(e)}"})
    # If x-db-test header is present and true, do not save
    if x_db_test and x_db_test.lower() == 'true':
        logger.info(f"[DB-CONFIG] Connection test passed for user {username} (not saved)")
        return JSONResponse(content={"success": True}, headers={"Content-Type": "application/json"})
    # Otherwise, save config
    save_user_db_config_to_db(user_id, body)
    logger.info(f"[DB-CONFIG] Settings saved for user {username}")
    return JSONResponse(content={"success": True}, headers={"Content-Type": "application/json"})

# --- Refactor connection logic ---
def get_connection(user_config=None):
    """Get a connection from the pool or direct using user config."""
    try:
        if user_config:
            logger.info(f"Connecting with user config: {user_config}")
            conn = psycopg2.connect(
                host=user_config['host'],
                database=user_config['database'],
                user=user_config['user'],
                password=user_config['password'],
                port=user_config['port']
            )
            return conn
        else:
            logger.info("Getting connection from pool...")
            conn = connection_pool.getconn()
            logger.info("Connection obtained from pool")
            return conn
    except Exception as e:
        logger.error(f"Failed to get connection: {str(e)}")
        raise Exception(f"Database connection failed: {str(e)}")

def release_connection(conn):
    """Release a connection back to the pool"""
    try:
        connection_pool.putconn(conn)
        logger.info("Connection released back to pool")
    except Exception as e:
        logger.error(f"Failed to release connection to pool: {str(e)}")

def clean_response_data(response_text: str) -> str:
    """Clean the response data to remove any unwanted content"""
    try:
        # First, remove any JSON blocks that might contain unwanted commands
        import re
        
        # Remove JSON blocks that contain MySQL or shell commands
        json_pattern = r'\{[^}]*(?:mysql|command|shell|username|password)[^}]*\}'
        response_text = re.sub(json_pattern, '', response_text, flags=re.IGNORECASE)
        
        # Remove any lines that contain unwanted patterns
        lines = response_text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
                
            # Skip lines that contain unwanted patterns
            unwanted_patterns = [
                'mysql',
                'shell command',
                'runshellcommand',
                'username',
                'password',
                'show code',
                'command:',
                'tool:',
                '```json',
                '```sql',
                '```'
            ]
            
            # Check if line contains any unwanted patterns
            if any(pattern in line.lower() for pattern in unwanted_patterns):
                continue
                
            # Skip lines that look like JSON
            if (line.startswith('{') and '}' in line) or (line.startswith('[') and ']' in line):
                continue
                
            clean_lines.append(line)
        
        cleaned_response = '\n'.join(clean_lines).strip()
        
        # If the response is empty after cleaning, return a default message
        if not cleaned_response:
            return "No data found."
        
        # Additional cleanup: remove any remaining unwanted text
        cleaned_response = re.sub(r'To list all tables.*?mysql.*?;', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        cleaned_response = re.sub(r'\{.*?"tool".*?\}', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove the specific pattern you mentioned
        cleaned_response = re.sub(r'To list all tables in your database using.*?mysql.*?command.*?```json.*?```', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        cleaned_response = re.sub(r'Command Declined.*?Command execution declined', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        cleaned_response = re.sub(r'mysql -u \[username\].*?SHOW TABLES.*?;', '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        
        return cleaned_response.strip()
        
    except Exception as e:
        logger.error(f"Error cleaning response data: {str(e)}")
        return response_text  # Return original if cleaning fails

async def execute_sql(query: str) -> pd.DataFrame:
    """Execute SQL query and return results as pandas DataFrame"""
    conn = None
    try:
        logger.info(f"Executing query: {query}")
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        results = cursor.fetchall()
        logger.info(f"Raw query results: {results}")  # Log raw results
        df = pd.DataFrame(results)
        logger.info(f"DataFrame created with columns: {df.columns.tolist()}")  # Log DataFrame info
        logger.info(f"DataFrame shape: {df.shape}")  # Log DataFrame dimensions
        logger.info(f"Query executed successfully. Found {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to execute query: {str(e)}")
        raise Exception(f"Failed to execute query: {str(e)}")
    finally:
        if conn:
            release_connection(conn)

async def get_database_schema() -> Dict[str, Any]:
    """Get complete database schema information"""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT 
                t.table_name,
                obj_description(pgc.oid) as table_description,
                pgc.reltuples as row_count
            FROM information_schema.tables t
            JOIN pg_class pgc ON pgc.relname = t.table_name
            WHERE t.table_schema = 'public'
        """)
        tables = [{
            "name": row[0],
            "description": row[1],
            "row_count": row[2]
        } for row in cursor.fetchall()]
        
        # Get columns for each table
        for table in tables:
            cursor.execute("""
                SELECT 
                    column_name,
                    data_type,
                    column_default,
                    is_nullable
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND table_schema = 'public'
            """, (table["name"],))
            table["columns"] = [{
                "name": row[0],
                "type": row[1],
                "default": row[2],
                "nullable": row[3]
            } for row in cursor.fetchall()]
        
        return {"tables": tables}
    except Exception as e:
        logger.error(f"Error getting database schema: {str(e)}")
        return {"tables": []}
    finally:
        if conn:
            release_connection(conn)

async def generate_sql_with_ollama(query: str, schema: Dict[str, Any]) -> str:
    """Generate SQL query from natural language using Ollama."""
    try:
        # Handle common queries with predefined patterns
        query_lower = query.lower().strip()
        
        # Pattern matching for common queries
        if any(phrase in query_lower for phrase in ['list tables', 'show tables', 'all tables', 'tables in database']):
            return "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"
        
        if any(phrase in query_lower for phrase in ['list users', 'show users', 'all users']):
            return "SELECT * FROM users LIMIT 10;"
        
        if any(phrase in query_lower for phrase in ['list sessions', 'show sessions', 'all sessions']):
            return "SELECT * FROM sessions LIMIT 10;"
        # Prepare the prompt for Ollama
        prompt = f"""You are a PostgreSQL expert. Generate ONLY a valid SQL query for the given question.

Database Schema:
{json.dumps(schema, indent=2)}

Question: {query}

CRITICAL RULES:
1. Return ONLY the SQL query, no explanations, no markdown, no additional text
2. Do not include any JSON, commands, or suggestions about MySQL or shell commands
3. Do not mention MySQL, shell commands, or any command-line tools
4. Do not provide instructions on how to use mysql command
5. Use exact column names from the schema above
6. For listing all tables, ALWAYS use: SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;
7. For table structure, use: SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'table_name';
8. Use proper PostgreSQL syntax
9. Always end with semicolon
10. Return ONLY the SQL statement, nothing else
11. Do not suggest alternative methods or tools

FORBIDDEN: Do not mention mysql, shell commands, command line tools, or provide JSON responses.

Common queries:
- List tables: SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
- Show table data: SELECT * FROM table_name LIMIT 10;
- Count rows: SELECT COUNT(*) FROM table_name;

SQL Query:"""

        logger.info(f"Sending prompt to Ollama: {prompt}")

        # Call Ollama API
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.text}")
            raise Exception("Failed to generate SQL query")
            
        result = response.json()
        sql_query = result.get('response', '').strip()
        logger.info(f"Raw Ollama response: {sql_query}")
        
        # Clean up the SQL query
        sql_query = re.sub(r'```sql|```', '', sql_query)
        sql_query = sql_query.split(';')[0] + ';'
        sql_query = sql_query.replace('`', '')
        
        # Fix common schema-related errors
        sql_query = sql_query.replace('schema_name', 'table_schema')
        
        # Remove any extra text that might be added by the AI model
        # Extract only the SQL query part
        lines = sql_query.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and (line.upper().startswith('SELECT') or line.upper().startswith('INSERT') or 
                        line.upper().startswith('UPDATE') or line.upper().startswith('DELETE') or
                        line.upper().startswith('WITH') or line.upper().startswith('CREATE') or
                        line.upper().startswith('ALTER') or line.upper().startswith('DROP')):
                clean_lines.append(line)
                break  # Take only the first valid SQL statement
        
        if clean_lines:
            sql_query = clean_lines[0]
            if not sql_query.endswith(';'):
                sql_query += ';'
        
        logger.info(f"Cleaned SQL query: {sql_query}")
        return sql_query
        
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        raise Exception(f"Failed to generate SQL query: {str(e)}")

@app.post("/chat2sql/execute")
async def execute_query_endpoint(request: Request, x_username: Optional[str] = Header(None)):
    try:
        # Parse request body
        body = await request.json()
        query = body.get('query')
        session_id = body.get('sessionId')
        username = x_username or 'default'
        user_config = get_user_db_config(x_username)
        if not user_config:
            return JSONResponse(status_code=400, content={"error": "Database is not configured. Please set it in Settings."})
        if not query:
            logger.error("No query provided in request")
            return JSONResponse(
                status_code=400,
                content={"detail": "Query parameter is required"},
                headers={"Content-Type": "application/json"}
            )
        logger.info(f"Processing new query: {query}")
        logger.info(f"Request body: {body}")
        # Get database schema
        schema = await get_database_schema_with_user_config(user_config)
        logger.info(f"Retrieved schema with {len(schema['tables'])} tables")
        # Generate SQL using Ollama
        sql_query = await generate_sql_with_ollama(query, schema)
        logger.info(f"Generated SQL for query '{query}': {sql_query}")
        # Execute query
        df = await execute_sql_with_user_config(sql_query, user_config)
        logger.info(f"Query executed. DataFrame shape: {df.shape}")
        # Format the data as a markdown table
        if not df.empty:
            logger.info(f"Creating table with {len(df)} rows and columns: {df.columns.tolist()}")
            table = "| " + " | ".join(df.columns) + " |\n"
            table += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
            for _, row in df.iterrows():
                row_values = [str(val) if val is not None else '' for val in row]
                table += "| " + " | ".join(row_values) + " |\n"
        else:
            logger.warning(f"No data found for query: {query}")
            table = "No data found."
        table = clean_response_data(table)
        # Save messages to database if session_id is provided
        if session_id:
            try:
                conn = get_connection()
                cursor = conn.cursor()
                
                # Save user message
                cursor.execute(
                    "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s) RETURNING id",
                    (session_id, 'user', query)
                )
                user_message_id = cursor.fetchone()[0]
                
                # Save AI response
                cursor.execute(
                    "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s) RETURNING id",
                    (session_id, 'assistant', table)
                )
                ai_message_id = cursor.fetchone()[0]
                
                conn.commit()
                logger.info(f"Saved chat2sql messages to database. User message ID: {user_message_id}, AI message ID: {ai_message_id}")
            except Exception as e:
                logger.error(f"Error saving messages to database: {str(e)}")
            finally:
                if conn:
                    release_connection(conn)
        
        # Prepare response
        response_data = {
            "data": table,
            "columns": df.columns.tolist()
        }
        
        logger.info(f"Returning response for query '{query}' with {len(df)} rows")
        return JSONResponse(
            content=jsonable_encoder(response_data),
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={"Content-Type": "application/json"}
        )

@app.get("/")
async def root():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = get_connection()
        release_connection(conn)
        return JSONResponse(
            content={"status": "healthy", "service": "sql-executor", "database": "connected"},
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            content={"status": "unhealthy", "service": "sql-executor", "error": str(e)},
            headers={"Content-Type": "application/json"}
        )

# Helper: get schema and execute_sql with user config
async def get_database_schema_with_user_config(user_config):
    conn = None
    try:
        conn = get_connection(user_config)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                t.table_name,
                obj_description(pgc.oid) as table_description,
                pgc.reltuples as row_count
            FROM information_schema.tables t
            JOIN pg_class pgc ON pgc.relname = t.table_name
            WHERE t.table_schema = 'public'
        """)
        tables = [{
            "name": row[0],
            "description": row[1],
            "row_count": row[2]
        } for row in cursor.fetchall()]
        for table in tables:
            cursor.execute("""
                SELECT 
                    column_name,
                    data_type,
                    column_default,
                    is_nullable
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND table_schema = 'public'
            """, (table["name"],))
            table["columns"] = [{
                "name": row[0],
                "type": row[1],
                "default": row[2],
                "nullable": row[3]
            } for row in cursor.fetchall()]
        return {"tables": tables}
    except Exception as e:
        logger.error(f"Error getting database schema: {str(e)}")
        return {"tables": []}
    finally:
        if conn:
            conn.close()

async def execute_sql_with_user_config(query: str, user_config) -> pd.DataFrame:
    conn = None
    try:
        logger.info(f"Executing query: {query}")
        conn = get_connection(user_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        results = cursor.fetchall()
        logger.info(f"Raw query results: {results}")
        df = pd.DataFrame(results)
        logger.info(f"DataFrame created with columns: {df.columns.tolist()}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Query executed successfully. Found {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to execute query: {str(e)}")
        raise Exception(f"Failed to execute query: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    logger.info("Starting SQL Executor API server...")
    uvicorn.run(app, host="0.0.0.0", port=5000)