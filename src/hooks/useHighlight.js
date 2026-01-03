import { useState, useEffect, useCallback } from 'react';

export function useHighlight(sendMessage) {
  const [selection, setSelection] = useState(null);
  const [buttonPosition, setButtonPosition] = useState(null);

  // Handle text selection
  const handleSelection = useCallback(() => {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    if (selectedText && selection.anchorOffset !== selection.focusOffset) {
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();

      setSelection({
        text: selectedText,
        range: range,
      });

      // Position the highlight button near the selection
      setButtonPosition({
        top: rect.top + window.scrollY - 40,
        left: rect.left + window.scrollX + rect.width / 2,
      });
    } else {
      setSelection(null);
      setButtonPosition(null);
    }
  }, []);

  // Clear selection
  const clearSelection = useCallback(() => {
    setSelection(null);
    setButtonPosition(null);
    window.getSelection().removeAllRanges();
  }, []);

  // Handle click outside to clear selection
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        !event.target.closest('.highlight-button') &&
        !event.target.closest('.chat-panel')
      ) {
        clearSelection();
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [clearSelection]);

  // Listen for text selection
  useEffect(() => {
    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, [handleSelection]);

  return {
    text: selection?.text || '',
    range: selection?.range || null,
    buttonPosition,
    isValid: !!selection?.text,
    clearSelection,
  };
}

export default useHighlight;