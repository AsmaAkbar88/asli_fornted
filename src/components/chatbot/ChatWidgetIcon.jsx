/**
 * ChatWidgetIcon - Floating chat icon button component
 * Displays unread badge and toggles chat panel
 */

import React, { forwardRef, useRef, useEffect } from 'react';
import '../../css/chatbot.css';

/**
 * ChatWidgetIcon component
 * @param {Object} props - Component props
 * @param {boolean} props.isOpen - Whether chat panel is currently open
 * @param {number} props.unreadCount - Number of unread messages
 * @param {Function} props.onClick - Click handler to toggle chat
 * @param {Function} props.onFocus - Focus handler
 * @param {React.Ref} ref - Forward ref
 */
const ChatWidgetIcon = forwardRef(({ isOpen, unreadCount, onClick, onFocus }, ref) => {
  const buttonRef = useRef(null);

  // Forward ref to internal button ref
  useEffect(() => {
    if (typeof ref === 'function') {
      ref(buttonRef.current);
    } else if (ref) {
      ref.current = buttonRef.current;
    }
  }, [ref]);

  // Handle keyboard interactions (Enter or Space to toggle)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onClick();
    }
  };

  return (
    <button
      ref={buttonRef}
      className={`chat-widget-icon ${isOpen ? 'open' : ''} ${unreadCount > 0 ? 'has-new-messages' : ''}`}
      onClick={onClick}
      onFocus={onFocus}
      onKeyDown={handleKeyDown}
      aria-label={isOpen ? 'Close chat' : 'Open chat'}
      aria-expanded={isOpen}
      aria-live="polite"
      type="button"
      tabIndex={0}
    >
      <svg
        className="chat-icon"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        focusable="false"
      >
        {/* Cute robot face */}
        <circle cx="12" cy="8" r="1" fill="currentColor" /> {/* Left eye */}
        <circle cx="16" cy="8" r="1" fill="currentColor" /> {/* Right eye */}
        <path d="M10 14s.8 2 4 2 4-2 4-2" stroke="currentColor" fill="none" /> {/* Smile */}
        <path d="M9 4h6a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z" stroke="currentColor" fill="none" /> {/* Body */}
        <path d="M7 10v1a3 3 0 0 0 6 0v-1" stroke="currentColor" fill="none" /> {/* Antenna */}
        <path d="M17 10v1a3 3 0 0 1-6 0v-1" stroke="currentColor" fill="none" /> {/* Antenna */}
      </svg>

      {unreadCount > 0 && (
        <span
          className={`chat-unread-badge ${unreadCount > 9 ? 'double-digit' : ''}`}
          aria-label={`${unreadCount} unread message${unreadCount !== 1 ? 's' : ''}`}
        >
          {unreadCount > 99 ? '99+' : unreadCount}
        </span>
      )}
    </button>
  );
});

ChatWidgetIcon.displayName = 'ChatWidgetIcon';

export default ChatWidgetIcon;