// User-related types
export interface User {
  id: string;
  username: string;
  name?: string;
  email?: string;
  role: 'admin' | 'user' | 'viewer';
}

// Chat-related types
export interface FileAttachment {
  name: string;
  type: string;
  size: number;
  url?: string;
  documentId?: string;
  status?: string;
  processingError?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean; // Added to support streaming messages
  fileAttachment?: FileAttachment; // Added to support file attachments
  isProcessingFile?: boolean; // Added to identify document processing messages
  conversationId?: string; // Added to identify the conversation this message belongs to
}

// Extended ChatMessage type with additional properties used throughout the application
export interface ExtendedChatMessage extends Omit<ChatMessage, 'role'> {
  role: 'user' | 'assistant' | 'system';
  sources?: any[]; // Sources for RAG responses
  useRag?: boolean; // Flag to indicate if RAG was used
  isContextTool?: boolean; // Flag to indicate if this is a context tool message
  isContextMessage?: boolean; // Flag to indicate this is a context message with user preferences
  isContextUpdate?: boolean; // Flag to indicate this is a context update (like shell command results)
  isProcessingOnly?: boolean; // Flag to indicate this is document processing, not message streaming
  isLoadingOnly?: boolean; // Flag to indicate this is just a loading indicator with no text
  documentId?: string; // Document ID for file attachments
  documentStatus?: string; // Status of document processing
  isSqlResult?: boolean; // Flag to indicate this is a SQL query result
  isSqlQuery?: boolean; // Flag to indicate this is a SQL query from the user
  predictor?: boolean; // Flag to indicate this is a predictor-related message
  predictions?: PredictionResult[]; // Array of prediction results
  downloadUrl?: string; // URL for downloading files
  fileName?: string; // Name of the file to download
  showDownloadButton?: boolean; // Flag to indicate if download button should be shown
  error?: string; // Error message for failed operations
  isUserCommand?: boolean; // Flag to indicate this is a user command in predictor mode
  isServerResponse?: boolean; // Flag to indicate this is a server response in predictor mode
  chat2sql?: boolean; // Flag to indicate this is a Chat2SQL message
  isUserMessage?: boolean; // Flag to indicate this is a user message in Chat2SQL mode
}

export interface PredictionResult {
  startpoint: string;
  endpoint: string;
  place_slack: number;
  cts_slack: number;
  predicted_route_slack: number;
  // Legacy fields for backward compatibility
  beginpoint?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: Date;
  last_message_timestamp: Date;
  is_active: boolean;
}

export interface ChatSessionResponse {
  session: ChatSession;
  messages: ChatMessage[];
  total: number;
}

// Dashboard metrics types
export interface UserStats {
  totalUsers: number;
  adminUsers: number;
  regularUsers: number;
  recentUsers: number;
}

export interface MessageStats {
  totalMessages: number;
  recentMessages: number;
  avgResponseTime: number;
  totalDocuments: number;
}

export interface LicenseUsage {
  totalLicenses: number;
  activeLicenses: number;
  expirationDate: string;
  daysRemaining: number;
}

export interface DashboardMetrics {
  userStats: UserStats;
  messageStats: MessageStats;
  licenseUsage: LicenseUsage;
}