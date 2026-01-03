/**
 * MessageInput - User input field with character counter
 */

import React, { useState, useEffect, useRef, forwardRef } from 'react';
import { MAX_QUERY_LENGTH, MIN_QUERY_LENGTH } from '../../utils/constants';
import '../../css/chatbot.css';

const MessageInput = forwardRef(
  ({ isLoading, disabled, onSend, placeholder, initialValue = '' }, ref
) => {
  const [inputValue, setInputValue] = useState(initialValue || '');
  const textareaRef = useRef(null);
  const characterCounterRef = useRef(null);

  useEffect(() => {
    setInputValue(initialValue || '');
  }, [initialValue]);

  useEffect(() => {
    if (typeof ref === 'function') {
      ref(textareaRef.current);
    } else if (ref) {
      ref.current = textareaRef.current;
    }
  }, [ref]);

  const characterCount = (inputValue || '').trim().length;
  const isOverLimit = characterCount > MAX_QUERY_LENGTH;
  const isUnderMin = characterCount > 0 && characterCount < MIN_QUERY_LENGTH;

  const handleSubmit = () => {
    const trimmedValue = (inputValue || '').trim();
    if (
      trimmedValue.length >= MIN_QUERY_LENGTH &&
      trimmedValue.length <= MAX_QUERY_LENGTH &&
      !isLoading
    ) {
      onSend(trimmedValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    } else if (e.key === 'Escape' && !e.shiftKey) {
      // Allow Escape to close chat panel
      // The parent component will handle this
    }
  };

  const handleChange = (e) => {
    const newValue = e.target.value;
    if (newValue.length <= MAX_QUERY_LENGTH) {
      setInputValue(newValue);
    }
  };

  return (
    <div className="chat-input-container">
      <div className="chat-input-area">
        <textarea
          ref={textareaRef}
          className="chat-input-textarea"
          value={inputValue}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || 'Type your question here...'}
          disabled={disabled || isLoading}
          aria-label="Type your message"
          aria-invalid={isOverLimit || (isUnderMin && characterCount > 0)}
          aria-describedby={isOverLimit || isUnderMin ? 'char-counter' : undefined}
          rows={3}
          maxLength={MAX_QUERY_LENGTH}
          tabIndex={0}
        />

        <div id="char-counter" className="chat-input-actions">
          <span
            ref={characterCounterRef}
            className={`chat-character-counter ${
              isOverLimit ? 'error' : isUnderMin ? 'warning' : ''
            }`}
            aria-live="polite"
          >
            {characterCount} / {MAX_QUERY_LENGTH}
          </span>

          <button
            type="button"
            className="chat-send-button"
            onClick={handleSubmit}
            disabled={
              disabled ||
              isLoading ||
              characterCount < MIN_QUERY_LENGTH ||
              characterCount > MAX_QUERY_LENGTH
            }
            aria-label="Send message"
            aria-disabled={
              disabled ||
              isLoading ||
              characterCount < MIN_QUERY_LENGTH ||
              characterCount > MAX_QUERY_LENGTH
            }
            tabIndex={0}
          >
            {isLoading ? (
              <span>Sending...</span>
            ) : (
              <>
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                  focusable="false"
                  style={{ width: '20px', height: '20px' }}
                >
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
                <span>Send</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
});

MessageInput.displayName = 'MessageInput';

export default MessageInput;