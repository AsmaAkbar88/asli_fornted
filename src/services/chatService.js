/**
 * Chat Service - API communication layer for chat functionality
 * Connects to the backend API for RAG-powered chat responses
 */

// Update the API_BASE_URL to point to your deployed backend
// Using window.env or default fallback since Docusaurus doesn't have process.env
const API_BASE_URL =
  (typeof window !== 'undefined' && window.env && window.env.REACT_APP_API_URL) ||
  (typeof process !== 'undefined' && process.env.REACT_APP_API_URL) ||
  'https://asmaakbar88.vercel.app/api';

/**
 * Sends a message to the backend API
 * @param {string} message - The user's message
 * @param {string|null} highlightedContext - Optional highlighted text context
 * @param {Object|null} context - Optional page context metadata
 * @returns {Promise<Object>} Response from the backend API
 */
export async function sendMessage(message, highlightedContext = null, context = null) {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: message,
        highlighted_text: highlightedContext,
        context: context
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Failed to get response');
    }

    const data = await response.json();

    return {
      id: data.response_id,
      content: data.content,
      timestamp: data.timestamp,
      generationTime: data.generation_time,
      confidenceScore: data.confidence_score
    };
  } catch (error) {
    console.error('Chat API error:', error);
    // Fallback response when backend is not available
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      return {
        id: Date.now().toString(),
        content: "Backend API is not available. Please check that your backend is deployed and the API URL is configured correctly.",
        timestamp: new Date().toISOString(),
        generationTime: 0,
        confidenceScore: 0
      };
    }
    throw error;
  }
}

/**
 * Health check function to verify backend connectivity
 * @returns {Promise<Object>} Health check response
 */
export async function healthCheck() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return await response.json();
  } catch (error) {
    console.error('Health check failed:', error);
    return { status: 'unhealthy', details: {} };
  }
}

// Additional API functions could be added here:
// export async function getChatHistory(sessionId) { ... }
// export async function createNewSession() { ... }
// export async function deleteMessage(messageId) { ... }