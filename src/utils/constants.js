/**
 * Constants for Chatbot Application
 */

// Message validation constants
export const MIN_QUERY_LENGTH = 1;
export const MAX_QUERY_LENGTH = 2000;
export const MAX_HIGHLIGHT_LENGTH = 500;
export const MAX_MESSAGES_PER_SESSION = 50;

// UI constants
export const CHAT_PANEL_WIDTH = 380;
export const CHAT_PANEL_HEIGHT = 500;
export const WIDGET_SIZE = 60;

// API constants
export const API_TIMEOUT = 30000; // 30 seconds
export const RETRY_ATTEMPTS = 3;
export const RETRY_DELAY = 1000; // 1 second

// Animation constants
export const ANIMATION_DURATION = 300; // ms
export const PULSE_INTERVAL = 2000; // ms

// Error messages
export const ERROR_MESSAGES = {
  QUERY_TOO_SHORT: `Query must be at least ${MIN_QUERY_LENGTH} character(s) long`,
  QUERY_TOO_LONG: `Query must not exceed ${MAX_QUERY_LENGTH} characters`,
  HIGHLIGHT_TOO_LONG: `Highlighted text must not exceed ${MAX_HIGHLIGHT_LENGTH} characters`,
  NETWORK_ERROR: 'Network error occurred. Please check your connection.',
  SERVER_ERROR: 'Server error occurred. Please try again later.',
  TIMEOUT_ERROR: 'Request timed out. Please try again.'
};

// Default configuration
export const DEFAULT_CONFIG = {
  enableTypingIndicator: true,
  enableMessagePersistence: true,
  enableHighlightFeature: true,
  maxRetries: RETRY_ATTEMPTS
};