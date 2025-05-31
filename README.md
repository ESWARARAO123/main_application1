# Platform Dashboard

A modern, responsive web application built with React, TypeScript, and Express that provides user management and monitoring capabilities with a beautiful, intuitive interface.

## 🚀 Features

- **Modern Design**
  - Beautiful, intuitive interface
  - Dark mode by default
  - Smooth animations and transitions
  - Responsive and mobile-friendly

- **Authentication System**
  - Secure login/logout functionality
  - Session-based authentication
  - Role-based access control (Admin/Viewer)

- **User Management**
  - User CRUD operations
  - Role management
  - Secure password handling
  - Responsive data tables

- **Performance**
  - Fast page loads
  - Optimized bundle size
  - Efficient state management
  - Real-time updates

## 🛠️ Technology Stack

### Frontend
- React 18
- TypeScript
- React Scripts (Build tool)
- TailwindCSS (Styling)
- React Router v6 (Routing)
- Axios (API calls)
- Heroicons (Icons)

### Backend
- Node.js
- Express
- PostgreSQL (Database)
- bcrypt (Password hashing)
- express-session (Session management)

## 📦 Project Structure

```
├── client/                    # Frontend application
│   ├── src/
│   │   ├── components/        # Reusable React components
│   │   ├── contexts/          # React context providers
│   │   ├── pages/             # Page components
│   │   ├── services/          # API services
│   │   └── config.ts          # Frontend configuration
│   ├── .env                   # Environment variables
│   └── build/                 # Build output directory
│
├── src/                       # Backend application
│   ├── routes/                # API routes
│   ├── services/              # Backend services
│   ├── scripts/               # Utility scripts
│   │   └── sql/               # SQL schema files
│   ├── database.js            # Database configuration
│   └── server.js              # Express server setup
│
├── conf/                      # Configuration files
│   └── config.ini             # Application configuration
│
├── documentation/             # Project documentation
│
├── memory-bank/               # Internal project knowledge
│
├── assets/                    # Static assets and data files
└── package.json               # Project dependencies
```

## 🏗️ Unified Architecture

The Platform Dashboard uses a unified architecture where the Express backend serves both the API endpoints and the React frontend static files:

### Key Architecture Points

1. **Single Server Deployment**
   - No separate frontend development server
   - All traffic goes through a single port
   - Simplified deployment and configuration

2. **API Structure**
   - All API endpoints are prefixed with `/api`
   - Clear separation between API calls and static assets
   - RESTful architecture for predictable interactions

3. **Frontend Static Files**
   - Built React app is served from `./client/build`
   - Path configurable via `static_root_path` in config.ini
   - Express serves all static assets (JS, CSS, images)

4. **SPA Routing Support**
   - Backend serves index.html for all non-API routes
   - Allows direct URL access to React routes
   - Client-side routing works seamlessly with server

5. **Centralized Configuration**
   - All settings in `conf/config.ini`
   - Frontend fetches configuration from backend
   - No hardcoded values in frontend code

### Flow Diagram

```
┌─────────────┐     HTTP Request     ┌─────────────────────────┐
│  Browser    │──────────────────────▶  Express Server         │
└─────────────┘                      │                         │
       ▲                             │  ┌─────────────────┐    │
       │                             │  │ API Routes      │    │
       │                             │  │ (/api/*)        │    │
       │                             │  └─────────────────┘    │
       │       HTTP Response         │                         │
       └─────────────────────────────│  ┌─────────────────┐    │
                                     │  │ Static Files    │    │
                                     │  │ (React Build)   │    │
                                     │  └─────────────────┘    │
                                     │                         │
                                     │  ┌─────────────────┐    │
                                     │  │ SPA Fallback    │    │
                                     │  │ Route           │    │
                                     │  └─────────────────┘    │
                                     └─────────────────────────┘
                                              │
                                              │ reads
                                              ▼
                                     ┌─────────────────────────┐
                                     │ config.ini              │
                                     └─────────────────────────┘
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn
- PostgreSQL database (or SQLite for simpler setup)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install backend dependencies:
```bash
npm install
```

3. Install frontend dependencies:
```bash
cd client
npm install
cd ..
```

4. Configure the application:
   - Create a `conf/config.ini` file (you can copy from `conf/config.ini.example` if it exists)
   - Update configuration values as needed
   - All configuration is managed from the backend's config.ini file

### Running the Application

#### Production Mode
1. Build and run the application:
```bash
# In the root directory
npm run deploy
```

#### Development Mode
1. Start the backend development server:
```bash
# In the root directory
npm run dev
```

2. In a separate terminal, build the frontend with watch mode:
```bash
# In the root directory
npm run dev:client
```

3. Access the application:
   - http://localhost:5634
   - (Port can be configured in config.ini)

### Configuration Options

#### Essential Configuration Properties

```ini
[server]
port = 5634                    # The port the server will listen on
static_root_path = ./client/build  # Path to frontend static files

[database]
type = postgresql              # Database type (postgresql or sqlite)
host = localhost               # Database host (for PostgreSQL)
port = 5432                    # Database port (for PostgreSQL)
name = copilot                 # Database name
user = postgres                # Database username (for PostgreSQL)
password = your_password       # Database password (for PostgreSQL)

[frontend]
title = Platform Dashboard     # Application title
api_url = /api                 # API URL prefix
default_theme = dark           # Default theme (dark, light)

[security]
cookie_secure = false          # Set to true for HTTPS environments
cookie_max_age = 86400000      # Session cookie lifetime in milliseconds (1 day)
secret_key = your_secret_key   # Secret for session encryption (change this!)
```

### Troubleshooting

#### Common Issues

1. **Cannot connect to database**
   - Verify database credentials in config.ini
   - Ensure the database server is running
   - For PostgreSQL, check that the database exists and user has proper permissions

2. **Frontend assets not loading**
   - Ensure you've built the frontend with `npm run build`
   - Check that the static_root_path in config.ini points to the correct directory
   - Verify that the build directory exists and contains the expected files

3. **API endpoints returning 404**
   - All API endpoints should be prefixed with `/api`
   - Check network requests in browser developer tools to ensure correct URLs

4. **Session not persisting**
   - Ensure cookie settings in config.ini are appropriate for your environment
   - For HTTPS environments, set cookie_secure=true
   - Check browser cookie storage to verify cookies are being set

5. **SPA routing issues (404 on page refresh)**
   - The server is configured to serve index.html for all non-API routes
   - If you see 404 errors, check the server.js SPA fallback route configuration

## 🔐 Authentication

### Default Credentials
- Username: admin
- Password: admin

### Authentication Flow
1. User submits login credentials
2. Backend validates and creates a session
3. Frontend stores authentication state in React context
4. Protected routes check authentication status
5. Session is maintained until logout

## 🔧 Configuration

### Application Configuration (conf/config.ini)
```ini
[server]
domain = 0.0.0.0
port = 5634
session_secret = your_secret_key
static_root_path = ./client/build

[database]
type = postgresql
host = localhost
port = 5432
name = copilot
user = postgres
password = your_password

[frontend]
title = Platform Dashboard
api_url = /api
default_theme = dark

[admin]
default_username = admin
default_password = admin

[security]
cookie_secure = false
cookie_max_age = 86400000
```

## 📱 Responsive Design Breakpoints

- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: > 1024px

## 🔒 Security Features

- Password hashing with bcrypt
- Session-based authentication
- CORS protection
- HTTP-only cookies
- Role-based access control
- Input validation
- Error handling

## 🛣️ API Routes

### Authentication
- POST /auth/login - User login
- POST /auth/logout - User logout
- GET /auth/me - Get current user

### User Management
- GET /users - List all users (Admin only)
- POST /users - Create new user (Admin only)
- DELETE /users/:id - Delete user (Admin only)

### Chat Interface
- POST /chatbot/sessions - Create new chat session
- GET /chatbot/sessions - List all chat sessions
- GET /chatbot/sessions/:id - Get specific chat session
- PUT /chatbot/sessions/:id - Update chat session
- DELETE /chatbot/sessions/:id - Delete chat session

## 💻 Development

### Code Organization
- Components follow single responsibility principle
- Context API for state management
- TypeScript for type safety
- Modular CSS with Tailwind
- Responsive design patterns

### Best Practices
- Mobile-first approach
- Progressive enhancement
- Semantic HTML
- Accessibility considerations
- Error boundary implementation
- Loading state handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

# Product Demo

## PDF Text Extraction Enhancement

This application includes an enhanced PDF text extraction mechanism using Python's `pdfplumber` library for improved accuracy in the RAG (Retrieval-Augmented Generation) pipeline.

### Features

- Improved PDF text extraction with page markers and layout preservation
- Maintains compatibility with existing document processing pipeline
- Graceful fallback to original extraction methods if Python extraction fails
- Configuration via `config.ini` file

### Setup

1. Make sure Python 3.x is installed on your system
2. Install pdfplumber:
   ```bash
   pip install --user pdfplumber
   ```
3. Configure the Python interpreter path in `conf/config.ini`:
   ```ini
   [python]
   interpreter = python3
   # Or path to your virtual environment: /path/to/venv/bin/python3
   ```

### How It Works

The enhanced text extraction processes PDF files in the following order:

1. Attempts extraction using Python's pdfplumber (best quality)
2. Falls back to LangChain's PDFLoader if Python extraction fails
3. Falls back to pdf-parse if LangChain extraction fails
4. Uses a placeholder message if all extraction methods fail

The extracted text maintains its original structure as much as possible, including page markers, which improves the context quality for RAG responses.

### Troubleshooting

If you encounter issues with PDF extraction:

1. Check the logs for specific error messages
2. Ensure pdfplumber is installed: `pip install --user pdfplumber`
3. Verify the Python interpreter path in `conf/config.ini`
4. Make sure the Python script has execute permissions: `chmod +x python/extract_text.py`

## License

[Your license information here] 