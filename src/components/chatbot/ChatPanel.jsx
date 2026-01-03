/**
 * ChatPanel - Main chat interface component
 * Contains MessageList, MessageInput, loading indicator, and controls
 */

import React, { forwardRef, useEffect, useRef } from 'react';
import { formatErrorMessage } from '../../utils/formatters';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import '../../css/chatbot.css';

/**
 * ChatPanel component
 * @param {Object} props - Component props
 * @param {boolean} props.isOpen - Whether panel is visible
 * @param {Array} props.messages - Chat messages
 * @param {boolean} props.isLoading - Loading state
 * @param {Error|string} props.error - Error message
 * @param {Function} props.onSend - Function to send message
 * @param {Function} props.onClear - Function to clear chat
 * @param {Function} props.onClose - Function to close panel
 * @param {Function} props.messagesEndRef - Ref for message list container
 * @param {Function} props.chatInputRef - Ref for input element
 * @param {string} props.highlightedText - Highlighted text context
 * @param {React.Ref} ref - Forward ref
 */
const ChatPanel = forwardRef(
  (
    {
      isOpen,
      messages,
      isLoading,
      error,
      onSend,
      onClear,
      onClose,
      messagesEndRef,
      chatInputRef,
      highlightedText,
    },
    ref
  ) => {
    const panelRef = useRef(null);
    const closeBtnRef = useRef(null);

    // Merge refs
    useEffect(() => {
      if (typeof ref === 'function') {
        ref(panelRef.current);
      } else if (ref) {
        ref.current = panelRef.current;
      }
    }, [ref]);

    // Focus management - prevent body scroll when open
    useEffect(() => {
      if (isOpen) {
        document.body.style.overflow = 'hidden';

        // Focus close button when panel opens
        if (closeBtnRef.current) {
          setTimeout(() => closeBtnRef.current.focus(), 100);
        }
      } else {
        document.body.style.overflow = '';
      }
      return () => {
        document.body.style.overflow = '';
      };
    }, [isOpen]);

    // Focus trap implementation
    useEffect(() => {
      if (!isOpen || !panelRef.current) return;

      const focusableElements = panelRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );

      const firstFocusable = focusableElements[0];
      const lastFocusable = focusableElements[focusableElements.length - 1];

      const handleKeyDown = (e) => {
        if (e.key === 'Tab') {
          if (e.shiftKey) {
            // Shift + Tab
            if (document.activeElement === firstFocusable) {
              e.preventDefault();
              lastFocusable.focus();
            }
          } else {
            // Tab
            if (document.activeElement === lastFocusable) {
              e.preventDefault();
              firstFocusable.focus();
            }
          }
        }
      };

      panelRef.current.addEventListener('keydown', handleKeyDown);
      return () => {
        panelRef.current?.removeEventListener('keydown', handleKeyDown);
      };
    }, [isOpen]);

    return (
      <div
        ref={panelRef}
        className={`chat-panel ${isOpen ? 'open' : 'closed'}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="chat-panel-title"
        aria-hidden={!isOpen}
      >
        <div className="chat-panel-header">
          <h2 id="chat-panel-title">Book Chat Assistant</h2>
          <button
            type="button"
            ref={closeBtnRef}
            className="chat-close-button"
            onClick={onClose}
            aria-label="Close chat"
            tabIndex={0}
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              focusable="false"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="chat-message-list" ref={messagesEndRef} role="log" aria-live="polite" aria-label="Chat messages">
          <MessageList messages={messages} />
        </div>

        {isLoading && (
          <div className="chat-loading-indicator" role="status" aria-live="polite">
            <div className="chat-loading-spinner" aria-hidden="true"></div>
            <span>Generating response...</span>
          </div>
        )}

        {error && (
          <div className="chat-error-message" role="alert" aria-live="assertive">
            <p>{formatErrorMessage(error)}</p>
            <button
              type="button"
              className="chat-error-button"
              onClick={() => {
                // Clear error and let user try again
                if (typeof onSend === 'function') {
                  // Error will be cleared on next send
                }
              }}
              aria-label="Dismiss error"
              tabIndex={0}
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="chat-input-area">
          <MessageInput
            ref={chatInputRef}
            isLoading={isLoading}
            onSend={onSend}
            disabled={!isOpen || isLoading}
            placeholder={
              highlightedText
                ? 'Ask about selected text...'
                : 'Type your question here...'
            }
            initialValue={highlightedText}
          />

          <div className="chat-actions">
            {messages.length > 0 && (
              <button
                type="button"
                className="chat-clear-button"
                onClick={onClear}
                disabled={isLoading}
                aria-label="Clear chat history"
                tabIndex={0}
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                  focusable="false"
                  style={{ width: '16px', height: '16px' }}
                >
                  <path d="M3 6h18" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1 2-2V8a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v8a2 2 0 0 1 2 2z" />
                </svg>
                Clear Chat
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }
);

ChatPanel.displayName = 'ChatPanel';

export default ChatPanel;