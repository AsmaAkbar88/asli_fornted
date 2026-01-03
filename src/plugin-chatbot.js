/**
 * Docusaurus plugin to inject chatbot
 */

export default function chatbotPlugin(context, options) {
  return {
    name: 'chatbot-plugin',
    injectHtmlTags({ content }) {
      return {
        headTags: [
          {
            tagName: 'link',
            attributes: {
              rel: 'stylesheet',
              href: '/css/chatbot.css',
            },
          },
        ],
        preBodyTags: [
          {
            tagName: 'div',
            attributes: {
              id: 'chatbot-widget-root',
            },
          },
        ],
      };
    },
  };
}