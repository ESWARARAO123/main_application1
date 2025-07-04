FROM node:20-slim

# Install Python 3.11 and required dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Copy Python requirements
COPY python/requirements.txt ./python/requirements.txt

# Setup Python virtual environment
RUN python3 -m venv /app/python/venv
ENV PATH="/app/python/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r python/requirements.txt

# Copy the rest of the application
COPY . .

# Build client application
WORKDIR /app/client
RUN npm install && npm run build

# Back to app root
WORKDIR /app

# Make Python scripts executable
RUN chmod +x python/RAG-MODULE/installvenv.sh

RUN ./python/RAG-MODULE/installvenv.sh

# Make entrypoint script executable
RUN chmod +x /app/Docker/docker-entrypoint.sh

# Expose the application port
EXPOSE 5640

# Set the entrypoint
ENTRYPOINT ["/app/Docker/docker-entrypoint.sh"]

# Command to run the application
CMD ["npm", "start"] 