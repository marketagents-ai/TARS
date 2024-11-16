# Discord Agent Memory Editor

A simple web interface for managing and editing Discord bot agent memories stored in a inverted index as a pickle file.

![image](https://github.com/user-attachments/assets/34e91da9-a170-45f7-9bc0-1848e3ba14a7)


## Overview

The Discord Agent Memory Editor is a web-based tool that allows you to manage, search, and edit memory entries for Discord bot agents. 

## Features

- **Memory Search**: Search through agent memories with filters for specific users
- **Memory Creation**: Add new memories for specific users
- **Memory Editing**: Edit existing memories through a modal interface
- **Real-time Validation**: Database validation with statistics display
- **User Management**: Integrated user list and filtering

## Interface Components

1. **Search Section**
   - Search bar with user filtering
   - Real-time search functionality

2. **Memory Creation**
   - User selection dropdown
   - Memory content input
   - Creation button

3. **Memory Display**
   - Scrollable list of memory entries
   - Relevance scores
   - Edit and delete capabilities

4. **Debug Information**
   - System status display
   - Database validation statistics

## Technical Details

### Frontend
- Built with HTML5, CSS3, and JavaScript
- Uses Bootstrap 5.1.3 for layout
- Custom CSS for cyberpunk styling
- Responsive design

### API Integration
- RESTful API endpoints
- User management
- Memory CRUD operations
- Database validation

## Project Structure
```
memory-editor/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── memory_manager.py
│   ├── models.py
│   └── api/
│       ├── __init__.py
│       └── endpoints.py
├── static/
│   └── index.html
└── requirements.txt
```

## Getting Started

1. Ensure all dependencies are installed
2. Run the application server
3. Access the web interface through your browser
4. Use the search and creation tools to manage memories

## API Endpoints

- `/api/users` - Get list of users
- `/api/validate` - Validate database and get statistics
- Additional CRUD endpoints for memory management
