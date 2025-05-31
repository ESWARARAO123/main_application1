# DATA Directory

This directory contains all data files used by the application. The contents of this directory are ignored by git (except for this README and .gitkeep files) to avoid storing large binary files in the repository.

## Structure

The DATA directory contains the following subdirectories:

- `documents/`: Stores uploaded document files organized by user ID and document ID
- `embeddings/`: Stores document embeddings and metadata
- `chroma_data/`: Contains ChromaDB vector database files
- `vector_store/`: Contains fallback vector storage when ChromaDB is not available

## Git Behavior

All contents of this directory are ignored by git except for:
- This README.md file
- The .gitkeep file (to ensure the directory itself is tracked)

This ensures that:
1. The directory structure is maintained in the repository
2. Large data files are not committed to git
3. Each developer/deployment can have their own local data

## Note

When deploying the application, ensure this directory exists and has the proper permissions for the application to write to it.
