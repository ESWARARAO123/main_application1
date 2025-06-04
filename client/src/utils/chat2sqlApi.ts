interface Chat2SqlResponse {
  data: string;  // Markdown formatted table
  columns: string[];
}

// Function to clean unwanted content from Chat2SQL responses
const cleanChat2SqlResponse = (data: string): string => {
  try {
    // Remove any JSON blocks that might contain MySQL commands or other unwanted text
    let cleanedData = data;
    
    // Remove JSON patterns that contain unwanted commands
    const jsonPattern = /\{[^}]*(?:mysql|command|shell|username|password|tool)[^}]*\}/gi;
    cleanedData = cleanedData.replace(jsonPattern, '');
    
    // Remove lines that contain unwanted patterns
    const lines = cleanedData.split('\n');
    const cleanLines = lines.filter(line => {
      const lowerLine = line.toLowerCase().trim();
      
      // Skip empty lines
      if (!lowerLine) return false;
      
      // Skip lines with unwanted patterns
      const unwantedPatterns = [
        'mysql',
        'shell command',
        'runshellcommand',
        'username',
        'password',
        'show code',
        'command:',
        'tool:',
        'to list all tables',
        '```json',
        '```sql',
        'command declined',
        'command execution declined',
        'mysql -u',
        'show tables'
      ];
      
      // Check if line contains any unwanted patterns
      if (unwantedPatterns.some(pattern => lowerLine.includes(pattern))) {
        return false;
      }
      
      // Skip lines that look like JSON
      if ((lowerLine.startsWith('{') && lowerLine.includes('}')) || 
          (lowerLine.startsWith('[') && lowerLine.includes(']'))) {
        return false;
      }
      
      return true;
    });
    
    let result = cleanLines.join('\n').trim();
    
    // Additional regex cleanup for specific patterns
    result = result.replace(/To list all tables in your database using[\s\S]*?mysql[\s\S]*?command[\s\S]*?```json[\s\S]*?```/gi, '');
    result = result.replace(/Command Declined[\s\S]*?Command execution declined/gi, '');
    result = result.replace(/mysql -u \[username\][\s\S]*?SHOW TABLES[\s\S]*?;/gi, '');
    result = result.replace(/⚬ Command execution declined/gi, '');
    
    return result.trim() || 'No data found.';
    
  } catch (error) {
    console.error('Error cleaning Chat2SQL response:', error);
    return data; // Return original if cleaning fails
  }
};

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
        
        // Clean the response data
        const cleanedData = {
          ...data,
          data: cleanChat2SqlResponse(data.data)
        };
        
        console.log('Cleaned data:', cleanedData);
        return cleanedData;
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
    
    // Clean the fallback response data as well
    const cleanedFallbackData = {
      ...fallbackData,
      data: cleanChat2SqlResponse(fallbackData.data)
    };
    
    console.log('Cleaned fallback data:', cleanedFallbackData);
    return cleanedFallbackData;
  } catch (error) {
    console.error('Error fetching chat2sql result:', error);
    throw error;
  }
}; 