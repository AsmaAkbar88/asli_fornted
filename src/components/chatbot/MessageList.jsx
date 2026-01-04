/**
 * MessageList - Display all messages in chat (ChatGPT-like UI)
 */

import React, { forwardRef } from 'react';
import { formatMessage, formatDateTime, formatUserName } from '../../utils/formatters';
import '../../css/chatbot.css';

const MessageList = forwardRef(({ messages }, ref) => {
  if (!messages || messages.length === 0) {
    return (
      <div className="chat-empty-state">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          aria-hidden="true"
          focusable="false"
          style={{ width: '64px', height: '64px', marginBottom: '16px', opacity: 0.5 }}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
          />
        </svg>
        <h3>No messages yet</h3>
        <p>Ask a question about book content to get started!</p>
      </div>
    );
  }

  return (
    <div
      ref={ref}
      className="chat-message-list"
      role="log"
      aria-live="polite"
      aria-label="Chat messages"
    >
      {messages.map((message) => (
        <div
          key={message.id}
          className={`chat-message ${message.role}`}
          role="article"
        >
          <div className="chat-message-content">
            {message.highlightedContext && (
              <blockquote className="chat-highlighted-context">
                <span className="chat-highlighted-context-label">Context:</span>
                <p className="chat-highlighted-text">{message.highlightedContext}</p>
              </blockquote>
            )}

            <div
              dangerouslySetInnerHTML={{ __html: formatMessage(message.content) }}
              role="presentation"
            />
          </div>

          {message.generationTime && message.role === 'assistant' && (
            <div
              className="chat-message-timestamp"
              aria-live="polite"
            >
              {message.generationTime.toFixed(2)}s
            </div>
          )}
        </div>
      ))}
    </div>
  );
});

MessageList.displayName = 'MessageList';

export default MessageList;