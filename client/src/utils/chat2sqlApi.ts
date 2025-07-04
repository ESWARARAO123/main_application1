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
    result = result.replace(/ ? Command execution declined/gi, '');
    
    return result.trim() || 'No data found.';
    
  } catch (error) {
    console.error('Error cleaning Chat2SQL response:', error);
    return data; // Return original if cleaning fails
  }
};

// Helper to get username from localStorage (assumes user info is stored as 'user')
function getUsername() {
  try {
    const user = localStorage.getItem('user');
    if (user) {
      const parsed = JSON.parse(user);
      return parsed.username || 'default';
    }
  } catch {}
  return 'default';
}

export const fetchChat2SqlResult = async (query: string, sessionId?: string): Promise<Chat2SqlResponse> => {
  try {
    console.log('?? Sending chat2sql request:', query, 'Session ID:', sessionId);

    // Get Chat2SQL URL from config.ini via backend API service
    let chat2sqlUrl: string;

    try {
      const configResponse = await fetch('/api/frontend-config', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Cache-Control': 'no-cache'
        }
      });

      if (!configResponse.ok) {
        throw new Error(`Configuration service returned ${configResponse.status}: ${configResponse.statusText}`);
      }

      if (!configResponse.headers.get('content-type')?.includes('application/json')) {
        throw new Error('Configuration service returned non-JSON response');
      }

      const config = await configResponse.json();

      if (!config.chat2sqlUrl) {
        throw new Error('Chat2SQL URL not found in configuration');
      }

      chat2sqlUrl = config.chat2sqlUrl;
      console.log('?? Using Chat2SQL URL from config service:', chat2sqlUrl);

    } catch (configError) {
      console.error('? Failed to load Chat2SQL configuration:', configError);
      throw new Error(`Chat2SQL configuration unavailable: ${configError instanceof Error ? configError.message : 'Unknown error'}`);
    }

    const requestBody = {
      query,
      sessionId,
      timestamp: Date.now()
    };
    console.log('?? Request body:', requestBody);

    const response = await fetch(`${chat2sqlUrl}/chat2sql/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'x-username': getUsername()
      },
      body: JSON.stringify(requestBody)
    });

    console.log('?? Response status from Python service:', response.status);
    console.log('?? Response headers:', Object.fromEntries(response.headers.entries()));

    if (!response.ok) {
      const errorText = await response.text();
      console.error('? Python service error response:', errorText);
      throw new Error(`Python Chat2SQL service failed: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    console.log('? Received data from Python service:', data);

    // Clean the response data
    const cleanedData = {
      ...data,
      data: cleanChat2SqlResponse(data.data)
    };

    console.log('?? Cleaned data:', cleanedData);
    return cleanedData;
  } catch (error) {
    console.error('Error fetching chat2sql result:', error);
    throw error;
  }
}; 