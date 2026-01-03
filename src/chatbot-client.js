/**
 * Docusaurus Client Module for Chatbot UI
 * Injects chatbot UI into all pages
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import ChatWidget from './components/chatbot/ChatWidget';

console.log('[Chatbot] Client module loading...');

// Immediately inject chatbot when module loads
function injectChatbot() {
  console.log('[Chatbot] Injecting chatbot...');

  // Create root container if it doesn't exist
  let container = document.getElementById('chatbot-widget-root');
  console.log('[Chatbot] container found:', !!container);

  if (!container) {
    container = document.createElement('div');
    container.id = 'chatbot-widget-root';
    document.body.appendChild(container);
    console.log('[Chatbot] container created');
  }

  // Create root and render ChatWidget
  try {
    console.log('[Chatbot] Creating React root...');
    const root = ReactDOM.createRoot(container);
    console.log('[Chatbot] Rendering ChatWidget...');
    root.render(React.createElement(ChatWidget));
    console.log('[Chatbot] Chatbot widget injected successfully');
  } catch (error) {
    console.error('[Chatbot] Failed to inject chatbot widget:', error);
  }
}

// Execute immediately
if (typeof window !== 'undefined') {
  injectChatbot();
}

console.log('[Chatbot] Client module loaded');