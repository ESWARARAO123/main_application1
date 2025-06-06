from fastapi import FastAPI, HTTPException, Request
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
#import requests
import os
from psycopg2 import pool
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Configuration
DB_CONFIG = {
    'host': '172.16.16.54',
    'database':'copilot',
    'user': 'postgres',
    'password': 'root',
    'port': '5432'
}

# Ollama Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

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

# Create a connection pool
connection_pool = pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=50,  # Increased from 20 to 50
    host=DB_CONFIG['host'],
    database=DB_CONFIG['database'],
    user=DB_CONFIG['user'],
    password=DB_CONFIG['password'],
    port=DB_CONFIG['port']
)

def get_connection():
    """Get a connection from the pool"""
    try:
        logger.info("Getting connection from pool...")
        conn = connection_pool.getconn()
        logger.info("Connection obtained from pool")
        return conn
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {str(e)}")
        raise Exception(f"Database connection failed: {str(e)}")

def release_connection(conn):
    """Release a connection back to the pool"""
    try:
        connection_pool.putconn(conn)
        logger.info("Connection released back to pool")
    except Exception as e:
        logger.error(f"Failed to release connection to pool: {str(e)}")

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
        prompt = f"""Given the following database schema:
{json.dumps(schema, indent=2)}

Generate a valid PostgreSQL SQL query for this question: {query}

Rules:
1. Only use tables that exist in the schema
2. Return valid PostgreSQL syntax
3. Do not include any explanations, only the SQL query
4. Do not use backticks or any special characters
5. Use proper table names from the schema
6. If asking about users, use the correct table name from the schema
7. Always use SELECT statement
8. Always include a WHERE clause if filtering data
9. Always use proper table aliases
10. Always use proper column names from the schema
11. For information_schema queries, use table_schema not schema_name
12. For listing tables, use: SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;
13. For showing table data, use: SELECT * FROM <table_name>;

SQL Query:"""

        logger.info(f"Sending prompt to Ollama: {prompt}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
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
        sql_query = sql_query.replace('schema_name', 'table_schema')
        logger.info(f"Cleaned SQL query: {sql_query}")
        return sql_query

    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        raise Exception(f"Failed to generate SQL query: {str(e)}")

@app.post("/chat2sql/execute")
async def execute_query_endpoint(request: Request):
    """Execute a natural language query and return the results."""
    try:
        # Parse request body
        body = await request.json()
        query = body.get('query')
        session_id = body.get('sessionId')  # Get session ID from request
        
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
        schema = await get_database_schema()
        logger.info(f"Retrieved schema with {len(schema['tables'])} tables")
        
        # Generate SQL using Ollama
        sql_query = await generate_sql_with_ollama(query, schema)
        logger.info(f"Generated SQL for query '{query}': {sql_query}")
        
        # Execute query
        df = await execute_sql(sql_query)
        logger.info(f"Query executed. DataFrame shape: {df.shape}")
        
        # Format the data as a markdown table
        if not df.empty:
            logger.info(f"Creating table with {len(df)} rows and columns: {df.columns.tolist()}")
            
            # Create markdown table
            table = "| " + " | ".join(df.columns) + " |\n"
            table += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
            
            # Add table rows
            for _, row in df.iterrows():
                # Convert all values to strings and handle None/null values
                row_values = [str(val) if val is not None else '' for val in row]
                table += "| " + " | ".join(row_values) + " |\n"
            
        else:
            logger.warning(f"No data found for query: {query}")
            table = "No data found."
        
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

if __name__ == "__main__":
    logger.info("Starting SQL Executor API server...")
    uvicorn.run(app, host="0.0.0.0", port=5000)