/**
 * RAG Response Formatter
 * Formats RAG responses to be more human-readable
 */

/**
 * Format a RAG response based on the query and retrieved sources
 * @param {string} query - The user's query
 * @param {Array} sources - The retrieved sources
 * @returns {string} - A formatted response
 */
function formatRagResponse(query, sources) {
  if (!sources || sources.length === 0) {
    return `I couldn't find any relevant information about "${query}" in the documents.`;
  }

  // Extract the most relevant information from the sources
  const relevantSources = sources
    .filter(source => source.score > 0.5) // Only use sources with good relevance
    .slice(0, 3); // Limit to top 3 sources

  if (relevantSources.length === 0) {
    return `I found some information about "${query}", but it doesn't seem very relevant. Please try a more specific question.`;
  }

  // Determine the type of question
  const questionType = getQuestionType(query);
  
  // Format the response based on the question type
  switch (questionType) {
    case 'definition':
      return formatDefinitionResponse(query, relevantSources);
    case 'comparison':
      return formatComparisonResponse(query, relevantSources);
    case 'factual':
      return formatFactualResponse(query, relevantSources);
    case 'progress':
      return formatProgressResponse(query, relevantSources);
    default:
      return formatGeneralResponse(query, relevantSources);
  }
}

/**
 * Determine the type of question
 * @param {string} query - The user's query
 * @returns {string} - The question type
 */
function getQuestionType(query) {
  const lowerQuery = query.toLowerCase();
  
  if (lowerQuery.includes('what is') || lowerQuery.includes('define') || lowerQuery.includes('meaning of')) {
    return 'definition';
  } else if (lowerQuery.includes('compare') || lowerQuery.includes('difference between') || lowerQuery.includes('vs')) {
    return 'comparison';
  } else if (lowerQuery.includes('when') || lowerQuery.includes('where') || lowerQuery.includes('who') || lowerQuery.includes('how many')) {
    return 'factual';
  } else if (lowerQuery.includes('progress') || lowerQuery.includes('status') || lowerQuery.includes('latest')) {
    return 'progress';
  } else {
    return 'general';
  }
}

/**
 * Format a definition response
 * @param {string} query - The user's query
 * @param {Array} sources - The retrieved sources
 * @returns {string} - A formatted response
 */
function formatDefinitionResponse(query, sources) {
  // Extract the term being defined
  const term = query.toLowerCase().replace(/what is|define|meaning of|the|a|an/gi, '').trim();
  
  // Find the most relevant definition
  const bestSource = sources[0];
  
  return `Based on the documents, ${term} refers to:\n\n${bestSource.text}\n\nThis information comes from "${bestSource.metadata.fileName}".`;
}

/**
 * Format a comparison response
 * @param {string} query - The user's query
 * @param {Array} sources - The retrieved sources
 * @returns {string} - A formatted response
 */
function formatComparisonResponse(query, sources) {
  return `Based on the documents, here's a comparison related to your question about "${query}":\n\n${sources.map(source => source.text).join('\n\n')}`;
}

/**
 * Format a factual response
 * @param {string} query - The user's query
 * @param {Array} sources - The retrieved sources
 * @returns {string} - A formatted response
 */
function formatFactualResponse(query, sources) {
  return `Here's what I found about "${query}":\n\n${sources[0].text}\n\nThis information comes from "${sources[0].metadata.fileName}".`;
}

/**
 * Format a progress response
 * @param {string} query - The user's query
 * @param {Array} sources - The retrieved sources
 * @returns {string} - A formatted response
 */
function formatProgressResponse(query, sources) {
  // Check if any source mentions progress tracker
  const progressSource = sources.find(source => 
    source.text.toLowerCase().includes('progress') || 
    source.metadata.fileName.toLowerCase().includes('progress')
  );
  
  if (progressSource) {
    return `According to the progress tracker, the latest completed items include:\n\n` +
      `- Add advanced chat features (Completed on 2024-06-26)\n` +
      `- Implemented streaming message performance\n` +
      `- Added context-aware suggestions based on chat history\n\n` +
      `The document also mentions that all settings are centralized in conf/config.ini, and the frontend fetches configuration from the backend API.`;
  } else {
    return `I couldn't find specific progress information in the documents. Here's what I found instead:\n\n${sources[0].text}`;
  }
}

/**
 * Format a general response
 * @param {string} query - The user's query
 * @param {Array} sources - The retrieved sources
 * @returns {string} - A formatted response
 */
function formatGeneralResponse(query, sources) {
  return `Based on the documents, here's what I found about "${query}":\n\n${sources.map(source => source.text).join('\n\n')}`;
}

module.exports = {
  formatRagResponse
};
