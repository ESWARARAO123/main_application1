# Docker Setup Instructions

This application uses Docker to run ChromaDB. Follow these instructions to install and set up Docker on your Windows machine.

## Installing Docker Desktop on Windows

1. **Download Docker Desktop**:
   - Visit [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
   - Click the "Download for Windows" button
   - Follow the installation instructions

2. **System Requirements**:
   - Windows 10 64-bit: Pro, Enterprise, or Education (Build 16299 or later)
   - Windows 11 64-bit
   - Hyper-V and Containers Windows features must be enabled
   - BIOS-level hardware virtualization support must be enabled in the BIOS settings

3. **Installation**:
   - Run the Docker Desktop Installer
   - Follow the installation wizard
   - When prompted, ensure the "Use WSL 2 instead of Hyper-V" option is selected (recommended)
   - Click "Ok" to start the installation

4. **Post-Installation**:
   - Docker Desktop will start automatically after installation
   - You may need to log out and log back in, or restart your computer
   - Docker Desktop icon will appear in the system tray

## Verifying Docker Installation

1. Open a Command Prompt or PowerShell window
2. Run the following command:
   ```
   docker --version
   ```
3. You should see output similar to:
   ```
   Docker version 20.10.x, build xxxxxxx
   ```
4. Also verify Docker Compose:
   ```
   docker-compose --version
   ```

## Starting Docker Desktop

1. If Docker Desktop is not running, you can start it from the Start menu
2. Search for "Docker Desktop" and click on it
3. Wait for Docker Desktop to start (the whale icon in the system tray will stop animating when it's ready)

## Troubleshooting

If you encounter issues with Docker:

1. **WSL 2 Installation**: If prompted to install WSL 2, follow the instructions provided by Docker Desktop
2. **Virtualization**: Ensure virtualization is enabled in your BIOS settings
3. **Windows Features**: Ensure Hyper-V and Containers Windows features are enabled
4. **Restart**: Sometimes a simple restart of Docker Desktop or your computer can resolve issues

## Using Docker with ChromaDB

Once Docker is installed and running:

1. Start Docker Desktop
2. Run the `start-chromadb.bat` file provided with this application
3. The ChromaDB server will start in a Docker container
4. You can verify it's running by visiting http://localhost:8000/api/v1/heartbeat in your browser

## Additional Resources

- [Docker Desktop Documentation](https://docs.docker.com/desktop/windows/)
- [Docker Get Started Guide](https://docs.docker.com/get-started/)
- [WSL 2 Installation Guide](https://docs.microsoft.com/en-us/windows/wsl/install)
