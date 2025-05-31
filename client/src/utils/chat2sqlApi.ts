interface Chat2SqlResponse {
  data: string;  // Markdown formatted table
  columns: string[];
}

export const fetchChat2SqlResult = async (query: string, sessionId?: string): Promise<Chat2SqlResponse> => {
  try {
    console.log('Sending chat2sql request:', query, 'Session ID:', sessionId);
    
    try {
      // First try the external Chat2SQL service
      const response = await fetch('http://localhost:5000/chat2sql/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',  // Prevent caching
          'Pragma': 'no-cache'
        },
        body: JSON.stringify({ 
          query,
          sessionId,  // Include session ID in request
          timestamp: Date.now()  // Add timestamp to make each request unique
        })
      });

      console.log('Response status from external service:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        console.log('Received data from external service for query:', query, data);
        return data;
      }
      
      console.warn('External Chat2SQL service failed, falling back to internal endpoint');
    } catch (externalError) {
      console.warn('Error connecting to external Chat2SQL service:', externalError);
      console.log('Falling back to internal endpoint');
    }
    
    // Fallback to our internal endpoint
    const fallbackResponse = await fetch('/api/chat2sql/execute', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
      },
      body: JSON.stringify({ 
        query,
        sessionId,
        timestamp: Date.now()
      }),
      credentials: 'include'  // Include cookies for authentication
    });
    
    console.log('Response status from internal fallback:', fallbackResponse.status);
    
    if (!fallbackResponse.ok) {
      const errorText = await fallbackResponse.text();
      console.error('Error response from internal fallback:', errorText);
      throw new Error(`HTTP error! status: ${fallbackResponse.status}, message: ${errorText}`);
    }
    
    const fallbackData = await fallbackResponse.json();
    console.log('Received data from internal fallback for query:', query, fallbackData);
    return fallbackData;
  } catch (error) {
    console.error('Error fetching chat2sql result:', error);
    throw error;
  }
}; 