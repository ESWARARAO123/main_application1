#!/bin/bash

# Navigate to the Docker directory
cd "$(dirname "$0")"

# Make sure the script is executable
chmod +x docker-entrypoint.sh

# Build and run the containers
docker-compose build
docker-compose up -d

echo "Application is starting..."
echo "You can access the application at http://localhost:5642"
echo "To view logs: docker-compose logs -f app"
echo "To stop: docker-compose down" 