export const generateId = () => {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

export const formatMessage = (content) => {
  if (!content) return '';
  return content.replace(/\n/g, '<br />');
};

export const formatDateTime = (isoTimestamp) => {
  if (!isoTimestamp) return '';
  try {
    const date = new Date(isoTimestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  } catch (error) {
    console.error('Error formatting date/time:', error);
    return '';
  }
};

export const formatErrorMessage = (error) => {
  if (!error) return 'An unknown error occurred';
  if (typeof error === 'string') return error;
  if (error.message) return error.message;
  return 'An error occurred. Please try again.';
};

export const formatUserName = (role) => {
  switch (role) {
    case 'user':
      return 'You';
    case 'assistant':
      return 'Book Assistant';
    case 'system':
      return 'System';
    default:
      return 'Unknown';
  }
};