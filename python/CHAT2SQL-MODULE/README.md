# CHAT2SQL-MODULE

## Project Overview
CHAT2SQL-MODULE is a FastAPI application that allows users to execute SQL queries using natural language input. The application leverages machine learning to convert user queries into SQL statements, interacts with a PostgreSQL database, and returns the results in a structured format.

## File Structure
```
CHAT2SQL-MODULE
├── backend.py          # FastAPI application handling requests and database interactions
├── requirements.txt    # Python dependencies for the project
├── Dockerfile          # Instructions to build a Docker image for the application
└── README.md           # Documentation for the project
```

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- PostgreSQL database
- Docker (optional, for containerization)

### Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd CHAT2SQL-MODULE
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application
To run the FastAPI application locally, execute:
```
uvicorn backend:app --host 0.0.0.0 --port 5000 --reload
```

### Using Docker
To build and run the application using Docker, follow these steps:
1. Build the Docker image:
   ```
   docker build -t chat2sql-module .
   ```

2. Run the Docker container:
   ```
   docker run -d -p 5000:5000 chat2sql-module
   ```

### API Endpoints
- **POST /chat2sql/execute**: Executes a natural language query and returns the results.
- **GET /**: Health check endpoint to verify the service status.

## Usage Example
To execute a query, send a POST request to `/chat2sql/execute` with a JSON body:
```json
{
  "query": "What are the names of all users?",
  "sessionId": "12345"
}
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.