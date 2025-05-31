/**
 * Enhances the system prompt with instructions for using the context reading tool
 * @param basePrompt The base system prompt
 * @returns Enhanced system prompt with context reading instructions
 */
export const enhancePromptWithContextReading = (basePrompt: string): string => {
  const contextReadingInstructions = `
You have access to a context reading tool that allows you to access the user's preferences and rules.
When you need to understand the user's preferences, you can use the read_context tool.

To use this tool, simply think:
"I should check if the user has any specific preferences or rules by using the read_context tool."

The context will contain:
- user_rules: The user's specific rules and preferences
- has_rules: Whether the user has defined any rules

Always respect the user's rules and preferences when responding.
`;

  return `${basePrompt}\n\n${contextReadingInstructions}`;
};
