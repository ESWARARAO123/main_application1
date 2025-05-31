# Platform Dashboard Documentation

## Overview

The Platform Dashboard is a modern, responsive web application designed to provide comprehensive user management, monitoring capabilities, and interactive features. It follows a client-server architecture with a React frontend and an Express.js backend, offering an intuitive and visually appealing interface for users.

## Purpose and Goals

The Platform Dashboard aims to:

1. Provide a centralized interface for users to monitor system status and runs
2. Enable seamless communication through an integrated chatbot
3. Offer robust user management with role-based access control
4. Deliver a customizable experience with theme preferences
5. Present a responsive design that works across various devices
6. Maintain security through proper authentication and authorization mechanisms

## Architecture

### High-Level Architecture

```
Platform Dashboard (Unified Architecture)
├── Frontend (React + TypeScript)
│   ├── Context-based State Management
│   ├── Component-based UI Architecture
│   └── Responsive Design
└── Backend (Express.js)
    ├── RESTful API Endpoints (/api prefix)
    ├── Static File Serving (frontend assets)
    ├── SPA Routing Support
    ├── SQLite Database Integration
    └── Authentication & Authorization
```

### Client-Server Communication

The application follows a unified architecture where:
- The backend serves both the API endpoints and static frontend files
- All API endpoints are prefixed with `/api` to distinguish them from static assets
- Frontend makes HTTP requests to backend endpoints
- Backend processes requests, interacts with the database, and returns responses
- Authentication is managed through sessions with HTTP-only cookies
- SPA routing is handled by the backend serving index.html for all non-API routes
- Configuration is centralized in a single config.ini file and provided to the frontend via API

## Technology Stack

### Frontend
- **React 18**: Core UI library
- **TypeScript**: Type-safe JavaScript
- **React Scripts**: Build tool for transpiling and bundling
- **TailwindCSS**: Utility-first CSS framework
- **React Router v6**: Client-side routing
- **Axios**: HTTP client for API requests
- **Heroicons**: SVG icon collection
- **Context API**: State management

### Backend
- **Node.js**: JavaScript runtime
- **Express**: Web server framework
- **SQLite3 (via better-sqlite3)**: Embedded database
- **bcrypt**: Password hashing
- **express-session**: Session management
- **cookie-parser**: Cookie handling
- **ini**: Configuration file parsing

## Project Structure

```
project-root/
├── client/                 # Frontend application
│   ├── src/
│   │   ├── components/     # Reusable React components
│   │   │   ├── AdminRoute.tsx      # Admin-only route protection
│   │   │   ├── Layout.tsx          # Main layout with sidebar/header
│   │   │   ├── LoadingScreen.tsx   # Loading indicator
│   │   │   ├── PrivateRoute.tsx    # Authentication protection
│   │   │   └── Sidebar.tsx         # Navigation sidebar
│   │   ├── contexts/       # React context providers
│   │   │   ├── AuthContext.tsx     # Authentication state
│   │   │   └── ThemeContext.tsx    # Theme preferences
│   │   ├── pages/          # Page components
│   │   │   ├── Chatbot.tsx         # Chatbot interface
│   │   │   ├── Dashboard.tsx       # Main dashboard
│   │   │   ├── Login.tsx           # Authentication page
│   │   │   ├── RunStatus.tsx       # Run monitoring
│   │   │   ├── Settings.tsx        # User preferences
│   │   │   └── UserManagement.tsx  # User administration
│   │   ├── services/       # API services
│   │   │   └── api.ts              # API client configuration
│   │   │   └── config.ts           # Configuration loading service
│   │   ├── styles/         # CSS styles
│   │   │   └── global.css          # Global styles, theme variables
│   │   ├── App.tsx         # Application routing
│   │   ├── config.ts       # Frontend configuration
│   │   ├── index.css       # Base styles
│   │   └── main.tsx        # Application entry point
│   ├── public/             # Static assets
│   ├── index.html          # HTML template
│   ├── package.json        # Frontend dependencies
│   └── build/              # Production build output
│
├── src/                    # Backend application
│   ├── routes/             # API route handlers
│   │   ├── auth.js         # Authentication endpoints
│   │   ├── chatbot.js      # Chatbot functionality
│   │   ├── config.js       # Frontend configuration endpoint
│   │   ├── runs.js         # Run management
│   │   ├── settings.js     # User preferences
│   │   └── users.js        # User management
│   ├── services/           # Backend services
│   │   └── config.js       # Configuration loading service
│   ├── database.js         # Database initialization and access
│   └── server.js           # Express server setup
│
├── conf/                   # Configuration
│   └── config.ini          # Centralized application configuration
│
├── data/                   # Data storage
│   └── app.db              # SQLite database
│
├── package.json            # Backend dependencies
└── README.md               # Project documentation
```

## Core Features and Implementation

### Authentication System

The application uses session-based authentication:

1. **Login Process:**
   - User submits credentials to `/auth/login`
   - Backend validates credentials and creates a session
   - Session ID is stored in an HTTP-only cookie
   - User data is stored in the AuthContext on the frontend

2. **Session Management:**
   - Sessions stored in SQLite database
   - Configurable session lifetime
   - Automatic session verification via middleware

3. **Authorization:**
   - Role-based access (admin vs. standard user)
   - Protected routes using PrivateRoute and AdminRoute components
   - Backend routes protected with isAuthenticated and isAdmin middleware

### User Management

Comprehensive user management with the following capabilities:

1. **User Operations:**
   - Creating new users (admin only)
   - Viewing and editing user profiles
   - Updating passwords
   - Deleting users (admin only)

2. **User Roles:**
   - Admin: Full system access
   - User: Limited access to personal settings and non-admin features

3. **Implementation:**
   - User data stored in SQLite database
   - Passwords hashed with bcrypt
   - Admin-only routes protected with middleware

### Theme System

Dynamic theme switching with persistent preferences:

1. **Available Themes:**
   - Dark: Dark blue interface
   - Light: Bright and clean interface
   - Midnight: Deep black theme with purple accents

2. **Implementation:**
   - ThemeContext manages current theme and theme switching
   - CSS variables for consistent theming across components
   - Theme preferences stored in localStorage
   - Server-side storage of theme preferences

### Dashboard & Monitoring

1. **Run Status:**
   - View status of various runs in different formats (list, flow, waffle)
   - Filter runs by type, status, and search term
   - Interactive elements to control runs (pause, resume, stop)

2. **Implementation:**
   - Data fetched from backend API
   - Interactive UI elements for filtering and controlling runs
   - Multiple view modes for different visualization preferences

### Chatbot Interface

Integrated chat interface for interacting with the system:

1. **Features:**
   - Text-based interaction
   - Message history with role-based grouping
   - Model Context Protocol (MCP) selection

2. **Implementation:**
   - Message state managed within component
   - Backend integration for message processing
   - Expandable for integration with various AI services

## Configuration

### Unified Configuration (config.ini)

The application is configured via a single config.ini file with the following sections:

1. **Server Settings:**
   - Host, port, protocol
   - Static file serving path
   - Session configuration

2. **Database Settings:**
   - Database type and path
   - Connection parameters

3. **Security Settings:**
   - Session secret
   - Cookie configuration
   - CORS settings

4. **Frontend Settings:**
   - Application name and theme
   - API URL and other frontend options
   - Default theme

5. **User Settings:**
   - Default admin credentials
   - User creation policies

### Configuration Example

```ini
[paths]
static_files = ./client/build

[server]
protocol = http
domain = localhost
port = 5634
static_root_path = ./client/build
session_secret = your-secret-key

[database]
type = sqlite
path = ./data/app.db

[frontend]
title = Platform Dashboard
api_url = /api
default_theme = dark

[security]
cookie_secure = false
cookie_max_age = 86400000
cookie_samesite = lax
secret_key = your-secret-key

[admin]
default_username = admin
default_password = admin
```

## API Endpoints

### Authentication

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|--------------|
| `/auth/login` | POST | User login | No |
| `/auth/logout` | POST | User logout | Yes |
| `/auth/me` | GET | Get current user info | Yes |

### User Management

| Endpoint | Method | Description | Auth Required | Admin Only |
|----------|--------|-------------|--------------|------------|
| `/users` | GET | List all users | Yes | Yes |
| `/users` | POST | Create new user | Yes | Yes |
| `/users/:id` | GET | Get user by ID | Yes | No* |
| `/users/profile` | PUT | Update user profile | Yes | No |
| `/users/password` | PUT | Update user password | Yes | No |
| `/users/:id` | DELETE | Delete user | Yes | Yes |

\* Users can only access their own information unless they're an admin

### Configuration

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|--------------|
| `/api/frontend-config` | GET | Get frontend configuration | No |

### Settings

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|--------------|
| `/settings/theme` | POST | Update user theme | Yes |
| `/settings/theme` | GET | Get user theme | Yes |
| `/settings/api-key` | POST | Save API key | Yes |
| `/settings/api-key` | GET | Get API key | Yes |

### Run Management

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|--------------|
| `/runs` | GET | List all runs | Yes |
| `/runs/:id` | GET | Get run by ID | Yes |
| `/runs/:id/start` | POST | Start a run | Yes |
| `/runs/:id/stop` | POST | Stop a run | Yes |
| `/runs/:id/pause` | POST | Pause a run | Yes |
| `/runs/:id/resume` | POST | Resume a run | Yes |

### Chatbot

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|--------------|
| `/chatbot/message` | POST | Send a message | Yes |
| `/chatbot/history` | GET | Get chat history | Yes |

## Database Schema

### Users Table

```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  username TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  role TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

### Sessions Table

```sql
CREATE TABLE sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  session_id TEXT UNIQUE NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users (id)
)
```

## Security Considerations

1. **Authentication & Authorization:**
   - Passwords hashed with bcrypt
   - HTTP-only cookies for session management
   - Role-based access control

2. **Data Protection:**
   - Input validation
   - SQL query parameter binding to prevent injection
   - Error handling to prevent information leakage

3. **Frontend Security:**
   - Protected routes
   - Secure API client configuration
   - Authentication state management

4. **Session Security:**
   - Configurable session lifetime
   - Session invalidation on logout
   - Secure cookie settings

## Responsive Design

The application is designed to work across various device sizes with responsive breakpoints:

- **Mobile:** < 640px
- **Tablet:** 640px - 1024px
- **Desktop:** > 1024px

Implementation includes:
- Responsive sidebar that collapses on smaller screens
- Adaptive layouts using Flexbox and Grid
- Conditional rendering of elements based on screen size
- Mobile-optimized forms and controls

## Development Workflow

### Local Development

1. **Configure the Application:**
   - Edit `conf/config.ini` with your local development settings
   - Make sure to set up database connections and other required configuration

2. **Start the Backend:**
   - Run `npm run dev` to start the Express server with nodemon for auto-reload
   - The server will read configuration from `conf/config.ini`

3. **Build the Frontend with Watch Mode:**
   - In a separate terminal, run `npm run dev:client` 
   - This builds the frontend and watches for changes
   - The backend will serve the built files from `client/build`

4. **Access the Application:**
   - Open your browser to `http://localhost:5634` (or the configured port)
   - The backend serves both the frontend assets and API endpoints

### Production Deployment

1. **Build the Application:**
   - Run `npm run build` to create a production build of the frontend

2. **Start the Production Server:**
   - Run `npm run start` or `npm run deploy` to build and start the server
   - The server will read configuration from `conf/config.ini`
   - All frontend assets are served by the backend

3. **Configuration:**
   - Update `conf/config.ini` with production settings
   - Ensure security settings like cookie security are properly configured

## Development Guidelines

1. **Code Organization:**
   - Follow component-based architecture
   - Use TypeScript for type safety
   - Maintain separation of concerns

2. **Styling:**
   - Use TailwindCSS utilities
   - Maintain theme consistency with CSS variables
   - Follow responsive design principles

3. **State Management:**
   - Use Context API for global state
   - Use component state for local concerns
   - Minimize prop drilling

4. **API Integration:**
   - Use the api.ts service for backend communication
   - Handle loading and error states
   - Implement proper error handling

5. **Testing:**
   - Write unit tests for components
   - Test API endpoints
   - Ensure responsive designs work

## Future Improvements

Potential areas for enhancement:

1. **Authentication:**
   - OAuth integration
   - Two-factor authentication
   - Password reset functionality

2. **Performance:**
   - Component code splitting
   - API response caching
   - Optimized bundle sizes

3. **Feature Expansion:**
   - Advanced analytics dashboard
   - Enhanced chatbot capabilities
   - Notification system
   - Team collaboration features

4. **Infrastructure:**
   - Docker containerization
   - CI/CD pipeline
   - Comprehensive logging

## Conclusion

The Platform Dashboard provides a robust foundation for building modern web applications with user management, monitoring capabilities, and interactive features. Its modular architecture allows for easy extension and customization to meet various use cases and requirements. 