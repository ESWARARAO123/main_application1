# Docker Deployment for ProductDemo

This directory contains Docker configuration files for deploying the ProductDemo application.

## Prerequisites

- Docker and Docker Compose installed on your host machine
- PostgreSQL database running on the host (accessible via host.docker.internal)
- Ollama service running on the host (accessible via host.docker.internal)

## Services

The Docker setup includes the following services:

1. **app** - Main application container running Node.js and Python
2. **chromadb** - Vector database for document embeddings
3. **redis** - Queue and cache service
4. **embedding-service** - Service for generating embeddings
5. **doc-workers** - Document processing workers
6. **text-processor** - Service for extracting text from documents
7. **mcp-orchestrator** - Service for executing MCP commands

## Configuration

The application uses a Docker-specific configuration file (`config.docker.ini`) that is copied to the correct location during container startup. This configuration is set up to:

- Connect to PostgreSQL and Ollama on the host machine
- Use containerized services for Redis, ChromaDB, etc.
- Configure all necessary paths and settings for Docker environment

## Getting Started

1. Make sure PostgreSQL is running on your host machine with the correct database created
2. Make sure Ollama is running on your host machine
3. Run the deployment script:

```bash
./run.sh
```

This will build and start all the Docker containers.

## Accessing the Application

Once the containers are running, you can access the application at:

http://localhost:5642

## Logs and Troubleshooting

To view logs from the main application container:

```bash
docker-compose logs -f app
```

To view logs from a specific service:

```bash
docker-compose logs -f [service-name]
```

## Stopping the Application

To stop all containers:

```bash
docker-compose down
```

To stop and remove all containers, networks, and volumes:

```bash
docker-compose down -v
```

## Data Persistence

The following data is persisted:

- PostgreSQL data (on host machine)
- ChromaDB data (in Docker volume)
- Redis data (in Docker volume)
- Application data in DATA directory (mounted from host)
- Application logs (mounted from host) 