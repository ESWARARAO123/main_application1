# ChromaDB Setup Instructions

This application uses ChromaDB for vector storage and retrieval. ChromaDB is used in client-server mode, which requires a running ChromaDB server.

## Prerequisites

- Docker installed and running on your machine
- Docker Compose installed (usually comes with Docker Desktop)

## Starting the ChromaDB Server

### Option 1: Using the provided batch file (Windows)

1. Make sure Docker Desktop is running
2. Run the `start-chromadb.bat` file by double-clicking it or running it from the command line

### Option 2: Using Docker Compose manually

1. Make sure Docker Desktop is running
2. Open a terminal/command prompt in the project root directory
3. Run the following command:
   ```
   docker-compose up -d chromadb
   ```

## Verifying the ChromaDB Server

To verify that the ChromaDB server is running:

1. Open a web browser
2. Navigate to http://localhost:8000/api/v1/heartbeat
3. You should see a response like `{"nanosecond heartbeat":1234567890}`

## Stopping the ChromaDB Server

To stop the ChromaDB server:

1. Open a terminal/command prompt in the project root directory
2. Run the following command:
   ```
   docker-compose down
   ```

## Troubleshooting

If you encounter issues with ChromaDB:

1. Check that Docker is running
2. Check that the ChromaDB container is running with `docker ps`
3. Check the logs of the ChromaDB container with `docker logs chromadb`
4. If needed, restart the ChromaDB container with:
   ```
   docker-compose restart chromadb
   ```

## Data Persistence

The ChromaDB data is stored in the `./DATA/chroma_data` directory. This ensures that your vector data persists even if the container is stopped or removed.
