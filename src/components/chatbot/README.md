# Book Chat Assistant

The Book Chat Assistant is an integrated chatbot that helps users navigate and understand the Physical AI and Humanoid Robotics documentation.

## Features

- **Documentation Search**: The chatbot can search through the book's documentation to find relevant information
- **Context Awareness**: Can respond to questions about highlighted text on the page
- **Responsive Design**: Works well on desktop, tablet, and mobile devices
- **Modern UI**: Clean, professional interface with smooth animations

## How It Works

The chatbot uses a simple search algorithm to find relevant content from the documentation based on user queries. It searches through:

- Tutorial Introduction
- Capstone Project: Physical AI System Integration
- Voice Processing
- Planning and Reasoning
- Navigation
- Object Detection
- Manipulation
- System Integration

## Using the Chatbot

1. **Access the Chatbot**: Click the chat icon in the bottom-right corner of the page
2. **Ask Questions**: Type questions about Physical AI, Humanoid Robotics, ROS 2, Gazebo, NVIDIA Isaac, VLA models, or system integration
3. **Highlight Text**: Select text on any page and click the floating "Ask" button to ask about specific content
4. **Clear Chat**: Use the "Clear Chat" button to reset the conversation

## Technical Implementation

- **Frontend**: React components with hooks for state management
- **Styling**: CSS with responsive design for all screen sizes
- **Data Source**: Documentation content embedded in the chat service
- **Search Algorithm**: Keyword-based search with relevance scoring

## Supported Topics

The chatbot can answer questions about:
- ROS 2 communication patterns
- Gazebo simulation
- NVIDIA Isaac platform
- Vision-Language-Action (VLA) models
- Physical AI system integration
- Voice processing for robotics
- Navigation and path planning
- Object detection and manipulation
- Safety and coordination in robotic systems

## Development

To enhance the chatbot further, you can:
1. Add more documentation content to the `DOCUMENTATION_CONTENT` object in `chatService.js`
2. Improve the search algorithm with more sophisticated text matching
3. Add integration with external APIs or databases for dynamic content
4. Implement conversation history and context persistence