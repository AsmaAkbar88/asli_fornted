/**
 * ChatWidget - Root component for chatbot UI
 * Manages overall chat state and coordinates all sub-components
 */

import React, { useEffect, useRef } from 'react';
import useChat from '../../hooks/useChat';
import useHighlight from '../../hooks/useHighlight';
import ChatWidgetIcon from './ChatWidgetIcon';
import ChatPanel from './ChatPanel';
import HighlightButton from './HighlightButton';
import '../../css/chatbot.css';

/**
 * ChatWidget component - Main root component
 */
function ChatWidget() {
  const chatState = useChat();
  const highlightState = useHighlight(chatState.sendMessage);

  // Refs for managing focus
  const widgetRef = useRef(null);
  const panelRef = useRef(null);

  // Handle highlight button "Ask" action
  const handleHighlightAsk = (highlightedText) => {
    chatState.setHighlighted(highlightedText);
  };

  // Keyboard shortcut to open/close (Ctrl/Cmd + /)
  useEffect(() => {
    const handleKeyPress = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === '/') {
        e.preventDefault();
        chatState.toggleChat();
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => {
      document.removeEventListener('keydown', handleKeyPress);
    };
  }, [chatState.toggleChat]);

  return (
    <>
      {/* Highlight Button - appears when text is selected */}
      {highlightState.isValid && highlightState.buttonPosition && (
        <HighlightButton
          text={highlightState.text}
          position={highlightState.buttonPosition}
          onAsk={handleHighlightAsk}
          onClear={highlightState.clearSelection}
        />
      )}

      {/* Floating Chat Widget Icon */}
      <ChatWidgetIcon
        ref={chatState.widgetIconRef}
        isOpen={chatState.isOpen}
        unreadCount={chatState.unreadCount}
        onClick={chatState.toggleChat}
        onFocus={() => {
          // Auto-scroll to bottom when focusing icon
          if (chatState.isOpen && chatState.messagesEndRef.current) {
            chatState.messagesEndRef.current.scrollTop =
              chatState.messagesEndRef.current.scrollHeight;
          }
        }}
      />

      {/* Chat Panel */}
      <ChatPanel
        ref={panelRef}
        isOpen={chatState.isOpen}
        messages={chatState.messages}
        isLoading={chatState.isLoading}
        error={chatState.error}
        onSend={chatState.sendMessage}
        onClear={chatState.clearChat}
        onClose={chatState.toggleChat}
        messagesEndRef={chatState.messagesEndRef}
        chatInputRef={chatState.chatInputRef}
        highlightedText={chatState.highlightedText}
      />
    </>
  );
}

/**
 * Lazy-loaded version for better performance
 * Docusaurus can load this dynamically
 */
export default ChatWidget;