import { api } from './api';

export interface ChatCompletionRequest {
  modelId: string;
  messages: {
    role: 'user' | 'assistant' | 'system';
    content: string;
  }[];
  options?: {
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    stream?: boolean;
  };
}

export interface ChatCompletionResponse {
  id: string;
  created: number;
  model: string;
  choices: {
    index: number;
    message: {
      role: 'assistant';
      content: string;
    };
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface StreamChunk {
  id: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta?: {
      role?: 'assistant';
      content?: string;
    };
    message?: {
      role: 'assistant';
      content: string;
    };
    finish_reason: string | null;
  }[];
}

export const aiChatService = {
  /**
   * Send a message to the Ollama API
   * @param request The chat completion request
   * @returns The chat completion response
   */
  sendChatCompletion: async (request: ChatCompletionRequest): Promise<ChatCompletionResponse> => {
    const transformedRequest = {
      model: request.modelId,
      messages: request.messages,
      ...(request.options && { options: request.options })
    };

    const response = await api.post('/ollama/chat', transformedRequest);

    if (!response.data || !response.data.choices) {
      const ollamaResponse = response.data;
      return {
        id: ollamaResponse.id || `chat-${Date.now()}`,
        created: ollamaResponse.created || Date.now(),
        model: ollamaResponse.model || request.modelId,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: ollamaResponse.message?.content || ollamaResponse.content || ''
            },
            finish_reason: ollamaResponse.finish_reason || 'stop'
          }
        ],
        usage: ollamaResponse.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
    }

    return response.data;
  },

  /**
   * Send a streaming message to the Ollama API
   * @param request The chat completion request
   * @param onChunk Callback for each received chunk
   * @param onComplete Callback when streaming is complete
   * @param onError Callback when an error occurs
   * @returns A function that can be called to abort the request
   */
  streamChatCompletion: async (
    request: ChatCompletionRequest,
    onChunk: (chunk: StreamChunk) => void,
    onComplete: () => void,
    onError: (error: any) => void
  ): Promise<() => void> => {
    // Create an AbortController to allow cancellation
    const controller = new AbortController();
    const signal = controller.signal;

    // Function to abort the request
    const abortRequest = () => {
      controller.abort();
      onComplete(); // Call onComplete to ensure proper cleanup
    };

    // Start the streaming process
    (async () => {
      try {
        const transformedRequest = {
          model: request.modelId,
          messages: request.messages,
          options: {
            ...request.options,
            stream: true
          }
        };

        const response = await fetch('/api/ollama/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(transformedRequest),
          signal // Add the abort signal
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        if (!response.body) {
          throw new Error('Response body is not readable');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          // Check if aborted
          if (signal.aborted) {
            console.log('Stream aborted by user');
            break;
          }

          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.trim() === '') continue;
            console.log('Raw chunk:', line); // Debug: Log raw chunk
            try {
              const chunk = JSON.parse(line);
              // Validate chunk structure
              if (chunk.choices?.[0] && (chunk.choices[0].delta || chunk.choices[0].message)) {
                onChunk(chunk);
              } else {
                console.warn('Skipping invalid chunk:', chunk);
              }
            } catch (e) {
              console.error('Error parsing stream chunk:', line, e);
            }
          }
        }

        // Handle final buffer
        if (buffer.trim() !== '' && !signal.aborted) {
          console.log('Raw final buffer:', buffer); // Debug: Log raw buffer
          try {
            const chunk = JSON.parse(buffer);
            // Validate chunk structure
            if (chunk.choices?.[0] && (chunk.choices[0].delta || chunk.choices[0].message)) {
              onChunk(chunk);
            } else {
              console.warn('Skipping invalid final chunk:', chunk);
            }
          } catch (e) {
            console.error('Error parsing final stream chunk:', buffer, e);
          }
        }

        // Only call onComplete if not aborted
        if (!signal.aborted) {
          onComplete();
        }
      } catch (error) {
        if (!signal.aborted) {
          onError(error);
        }
      }
    })();

    // Return the abort function
    return abortRequest;
  },

  /**
   * Get available AI models
   * @returns Array of available models
   */
  getAvailableModels: async () => {
    const response = await api.get('/ollama/models/active');
    return response.data.models;
  }
};

export default aiChatService;