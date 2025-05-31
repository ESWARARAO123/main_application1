@echo off
echo Starting ChromaDB server...
docker-compose up -d chromadb
echo.
echo ChromaDB server started at http://localhost:8000
echo.
echo To stop the server, run: docker-compose down
echo.
