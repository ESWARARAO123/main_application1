import { api } from './api';

export interface MCPChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface MCPChatCompletionRequest {
  modelId: string;
  messages: MCPChatMessage[];
  options?: {
    stream?: boolean;
    temperature?: number;
    max_tokens?: number;
  };
  mcpClientId: string;
  mcpServer: {
    host: string;
    port: number;
  };
}

export interface MCPChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string | null;
  }[];
}

export interface StreamChunk {
  id: string;
  object?: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta?: {
      content?: string;
    };
    message?: {
      role?: string;
      content?: string;
    };
    finish_reason: string | null;
  }[];
}

const mcpChatService = {
  /**
   * Send a chat message to the MCP proxy and get a completion
   */
  sendChatCompletion: async (request: MCPChatCompletionRequest): Promise<MCPChatCompletionResponse> => {
    const response = await api.post('/mcp/chat', request);
    return response.data;
  },

  /**
   * Stream a chat message through the MCP proxy
   */
  streamChatCompletion: async (
    request: MCPChatCompletionRequest,
    onChunk: (chunk: StreamChunk) => void,
    onComplete: () => void,
    onError: (error: Error) => void
  ): Promise<() => void> => {
    let fullResponse = '';
    const abortController = new AbortController();
    const signal = abortController.signal;

    let isAborting = false;

    try {
      // Make sure streaming is enabled
      if (!request.options || !request.options.stream) {
        request.options = { ...request.options, stream: true };
      }

      // Make a request to our backend API
      const response = await fetch('/api/mcp/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: signal
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to stream chat completion');
      }

      // Get the reader for the response body stream
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('ReadableStream not supported');
      }

      // Read the stream
      const decoder = new TextDecoder();
      let done = false;

      while (!done && !isAborting) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;

        if (done) {
          break;
        }

        // Decode the chunk and process it
        const text = decoder.decode(value);
        
        // Process the chunk based on SSE format
        const lines = text.trim().split('\n\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = line.slice(6); // Remove 'data: '
              
              // Skip empty data or [DONE]
              if (data === '' || data === '[DONE]') continue;
              
              const parsedData = JSON.parse(data);
              
              // Handle the SSE data structure which might differ from the standard OpenAI format
              let chunk: StreamChunk;
              
              if (parsedData.choices) {
                // Standard OpenAI format
                chunk = parsedData;
              } else if (parsedData.content) {
                // Simplified format with just content
                chunk = {
                  id: `chunk-${Date.now()}`,
                  created: Date.now(),
                  model: request.modelId || "",
                  choices: [{
                    index: 0,
                    delta: { content: parsedData.content },
                    message: { 
                      role: "assistant", 
                      content: parsedData.content 
                    },
                    finish_reason: null
                  }]
                };
              } else {
                // Unknown format, create a default
                chunk = {
                  id: `chunk-${Date.now()}`,
                  created: Date.now(),
                  model: request.modelId || "",
                  choices: [{
                    index: 0,
                    delta: { content: JSON.stringify(parsedData) },
                    message: { 
                      role: "assistant", 
                      content: JSON.stringify(parsedData) 
                    },
                    finish_reason: null
                  }]
                };
              }
              
              // Call the onChunk callback
              onChunk(chunk);
              
              // Accumulate the response
              const content = chunk.choices?.[0]?.delta?.content || chunk.choices?.[0]?.message?.content || '';
              fullResponse += content;
              
              // Check if this is the last chunk
              if (parsedData.done) {
                done = true;
                break;
              }
            } catch (e) {
              console.error('Error parsing chunk:', e, line);
            }
          }
        }
      }

      // Call onComplete when we're done, unless we're aborting
      if (!isAborting) {
        onComplete();
      }
    } catch (error: any) {
      // Only call onError if we're not manually aborting
      if (!isAborting && error.name !== 'AbortError') {
        onError(error);
      }
    }

    // Return a function to abort the stream
    return () => {
      isAborting = true;
      abortController.abort();
      onComplete();  // Call onComplete when manually aborted
    };
  }
};

export default mcpChatService;
export type { MCPChatCompletionRequest as ChatCompletionRequest, MCPChatCompletionResponse as ChatCompletionResponse }; 