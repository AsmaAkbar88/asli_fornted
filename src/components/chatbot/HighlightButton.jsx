/**
 * HighlightButton - Button that appears when text is selected
 * Allows user to ask about selected text
 */

import React from 'react';
import '../../css/chatbot.css';

/**
 * HighlightButton component
 * @param {Object} props - Component props
 * @param {string} props.text - Selected text
 * @param {Object} props.position - Position for the button
 * @param {Function} props.onAsk - Function to handle ask action
 * @param {Function} props.onClear - Function to clear selection
 */
const HighlightButton = ({ text, position, onAsk, onClear }) => {
  if (!position || !text) {
    return null;
  }

  const handleClick = () => {
    onAsk(text);
    onClear();
  };

  return (
    <button
      className="highlight-button"
      style={{
        top: `${position.top}px`,
        left: `${position.left}px`,
      }}
      onClick={handleClick}
      aria-label="Ask about selected text"
      type="button"
    >
      Ask Chatbot
    </button>
  );
};

HighlightButton.displayName = 'HighlightButton';

export default HighlightButton;