import { useState, useCallback, useRef, useEffect } from 'react';
import { sendMessage as apiSendMessage } from '../services/chatService';
import { generateId, formatDateTime } from '../utils/formatters';
import { MIN_QUERY_LENGTH, MAX_QUERY_LENGTH, MAX_HIGHLIGHT_LENGTH, MAX_MESSAGES_PER_SESSION } from '../utils/constants';

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isOpen, setIsOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const [highlightedText, setHighlightedText] = useState(null);

  const messagesEndRef = useRef(null);
  const chatInputRef = useRef(null);
  const widgetIconRef = useRef(null);
  const wasOpenRef = useRef(false);

  const scrollToBottom = useCallback((behavior = 'auto') => {
    if (messagesEndRef.current) {
      // Use smooth scrolling for better UX
      messagesEndRef.current.scrollTo({
        top: messagesEndRef.current.scrollHeight,
        behavior: behavior
      });
    }
  }, []);

  const sendMessage = useCallback(
    async (query, highlightedContext = null) => {
      if (!query || query.trim().length < MIN_QUERY_LENGTH) {
        setError(`Query must be at least ${MIN_QUERY_LENGTH} characters long`);
        return;
      }
      if (query.trim().length > MAX_QUERY_LENGTH) {
        setError(`Query must not exceed ${MAX_QUERY_LENGTH} characters`);
        return;
      }
      if (highlightedContext && highlightedContext.length > MAX_HIGHLIGHT_LENGTH) {
        setError(`Highlighted text must not exceed ${MAX_HIGHLIGHT_LENGTH} characters`);
        return;
      }
      setError(null);

      const userMessage = {
        id: generateId(),
        role: 'user',
        content: query.trim(),
        timestamp: new Date().toISOString(),
        highlightedContext: highlightedContext || null,
      };

      setMessages((prev) => {
        const newMessages = [...prev, userMessage];
        if (newMessages.length > MAX_MESSAGES_PER_SESSION) {
          return newMessages.slice(newMessages.length - MAX_MESSAGES_PER_SESSION);
        }
        return newMessages;
      });
      setIsLoading(true);

      try {
        const responseData = await apiSendMessage(userMessage.content, highlightedContext || null);

        const assistantMessage = {
          id: responseData.id || generateId(),
          role: 'assistant',
          content: responseData.content,
          timestamp: responseData.timestamp || new Date().toISOString(),
          generationTime: responseData.generationTime,
          confidenceScore: responseData.confidenceScore,
        };

        setMessages((prev) => {
          const newMessages = [...prev, assistantMessage];
          if (newMessages.length > MAX_MESSAGES_PER_SESSION) {
            return newMessages.slice(newMessages.length - MAX_MESSAGES_PER_SESSION);
          }
          return newMessages;
        });

        if (!isOpen) {
          setUnreadCount((prev) => prev + 1);
        }
      } catch (err) {
        const errorMessage = {
          id: generateId(),
          role: 'system',
          content: err.message || 'Failed to send message. Please try again.',
          timestamp: new Date().toISOString(),
          error: err.message,
        };

        setMessages((prev) => {
          const newMessages = [...prev, errorMessage];
          if (newMessages.length > MAX_MESSAGES_PER_SESSION) {
            return newMessages.slice(newMessages.length - MAX_MESSAGES_PER_SESSION);
          }
          return newMessages;
        });
        setError(err.message || 'Failed to send message. Please try again.');
      } finally {
        setIsLoading(false);
      }
    },
    [isOpen, MIN_QUERY_LENGTH, MAX_QUERY_LENGTH, MAX_HIGHLIGHT_LENGTH, MAX_MESSAGES_PER_SESSION]
  );

  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
    setUnreadCount(0);
  }, []);

  const toggleChat = useCallback(() => {
    setIsOpen((prev) => {
      const newState = !prev;
      wasOpenRef.current = prev;

      if (newState && !prev) {
        setUnreadCount(0);
        setTimeout(() => {
          if (chatInputRef.current) {
            chatInputRef.current.focus();
          }
        }, 100);
      }

      if (!newState && prev && widgetIconRef.current) {
        setTimeout(() => {
          widgetIconRef.current.focus();
        }, 300);
      }

      return newState;
    });
  }, []);

  const setHighlighted = useCallback((text) => {
    setHighlightedText(text);
  }, []);

  const handleKeyDown = useCallback(
    (event) => {
      if (!isOpen) return;
      if (event.key === 'Escape') {
        toggleChat();
      }
    },
    [isOpen, toggleChat]
  );

  useEffect(() => {
    if (messages.length > 0) {
      // Use setTimeout to ensure DOM has updated before scrolling
      setTimeout(() => {
        scrollToBottom('smooth');
      }, 50);
    }
  }, [messages, scrollToBottom]);

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, handleKeyDown]);

  useEffect(() => {
    if (wasOpenRef.current && !isOpen && messages.length > 0) {}
  }, [isOpen, messages.length]);

  return {
    messages,
    isLoading,
    error,
    isOpen,
    unreadCount,
    highlightedText,
    sendMessage,
    clearChat,
    toggleChat,
    setHighlighted,
    messagesEndRef,
    chatInputRef,
    widgetIconRef,
  };
}

export default useChat;