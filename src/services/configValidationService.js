/**
 * Configuration Validation Service
 * Validates and provides centralized access to all configuration values
 * Ensures no hardcoded values are used throughout the application
 */

const config = require('../utils/config');
const { logger } = require('../utils/logger');

class ConfigValidationService {
  constructor() {
    this.validatedConfig = null;
    this.validationErrors = [];
    this.warnings = [];
  }

  /**
   * Validate all configuration sections and return a normalized config object
   */
  async validateConfiguration() {
    this.validationErrors = [];
    this.warnings = [];

    try {
      const validatedConfig = {
        server: this.validateServerConfig(),
        database: this.validateDatabaseConfig(),
        docker: this.validateDockerConfig(),
        redis: this.validateRedisConfig(),
        ollama: this.validateOllamaConfig(),
        ai: this.validateAIConfig(),
        documentProcessing: this.validateDocumentProcessingConfig(),
        imageProcessing: this.validateImageProcessingConfig(),
        mcp: this.validateMCPConfig(),
        chat2sql: this.validateChat2SqlConfig()
      };

      this.validatedConfig = validatedConfig;

      // Log validation results
      if (this.validationErrors.length > 0) {
        logger.error('Configuration validation failed:', this.validationErrors);
        throw new Error(`Configuration validation failed: ${this.validationErrors.join(', ')}`);
      }

      if (this.warnings.length > 0) {
        logger.warn('Configuration warnings:', this.warnings);
      }

      logger.info('Configuration validation completed successfully');
      return validatedConfig;

    } catch (error) {
      logger.error('Configuration validation error:', error);
      throw error;
    }
  }

  /**
   * Validate server configuration
   */
  validateServerConfig() {
    // Try to read from [server] section first (new format), fallback to [app] section (legacy)
    const serverConfig = config.getSection('server');
    const appConfig = config.getSection('app');

    const validated = {
      protocol: serverConfig.protocol || 'http',
      domain: serverConfig.domain || appConfig.host || '0.0.0.0',
      port: this.validatePort('server.port', serverConfig.port || appConfig.port, 5642),
      staticRootPath: serverConfig.static_root_path || './client/build',
      serveFromSubPath: this.validateBoolean('server.serve_from_sub_path', serverConfig.serve_from_sub_path, false)
    };

    // Construct full URL
    validated.baseUrl = `${validated.protocol}://${validated.domain}:${validated.port}`;

    return validated;
  }

  /**
   * Validate database configuration
   */
  validateDatabaseConfig() {
    const dbConfig = config.getSection('database');
    
    return {
      type: this.validateRequired('database.database-type', dbConfig['database-type'], 'postgres'),
      host: this.validateRequired('database.database-host', dbConfig['database-host'], 'localhost'),
      port: this.validatePort('database.database-port', dbConfig['database-port'], 5432),
      name: this.validateRequired('database.database-name', dbConfig['database-name'], 'copilot'),
      user: this.validateRequired('database.database-user', dbConfig['database-user'], 'postgres'),
      password: this.validateRequired('database.database-password', dbConfig['database-password'], ''),
      maxConnections: this.validateNumber('database.max_connections', dbConfig.max_connections, 100),
      ssl: this.validateBoolean('database.ssl', dbConfig.ssl, false)
    };
  }

  /**
   * Validate Docker services configuration
   */
  validateDockerConfig() {
    const dockerConfig = config.getSection('docker');
    
    return {
      chromadb: {
        protocol: this.validateRequired('docker.chromadb_protocol', dockerConfig.chromadb_protocol, 'http'),
        host: this.validateRequired('docker.chromadb_host', dockerConfig.chromadb_host, 'localhost'),
        port: this.validatePort('docker.chromadb_port', dockerConfig.chromadb_port, 8001),
        get url() { return `${this.protocol}://${this.host}:${this.port}`; }
      },
      redis: {
        host: this.validateRequired('docker.redis_host', dockerConfig.redis_host, 'localhost'),
        port: this.validatePort('docker.redis_port', dockerConfig.redis_port, 6379)
      },
      postgres: {
        host: this.validateRequired('docker.postgres_host', dockerConfig.postgres_host, 'localhost'),
        port: this.validatePort('docker.postgres_port', dockerConfig.postgres_port, 5432)
      }
    };
  }

  /**
   * Validate Redis configuration
   */
  validateRedisConfig() {
    const redisConfig = config.getSection('redis');
    
    return {
      host: this.validateRequired('redis.host', redisConfig.host, 'localhost'),
      port: this.validatePort('redis.port', redisConfig.port, 6379),
      password: redisConfig.password || '',
      maxRetries: this.validateNumber('redis.max_retries', redisConfig.max_retries, 3),
      retryDelay: this.validateNumber('redis.retry_delay', redisConfig.retry_delay, 100),
      connectionTimeout: this.validateNumber('redis.connection_timeout', redisConfig.connection_timeout, 5000)
    };
  }

  /**
   * Validate Ollama configuration
   */
  validateOllamaConfig() {
    const ollamaConfig = config.getSection('ollama');
    
    const validated = {
      protocol: this.validateRequired('ollama.protocol', ollamaConfig.protocol, 'http'),
      host: this.validateRequired('ollama.host', ollamaConfig.host, 'localhost'),
      port: this.validatePort('ollama.port', ollamaConfig.port, 11434),
      connectionTimeout: this.validateNumber('ollama.connection_timeout', ollamaConfig.connection_timeout, 30000),
      requestTimeout: this.validateNumber('ollama.request_timeout', ollamaConfig.request_timeout, 120000)
    };

    // Construct full URL
    validated.baseUrl = `${validated.protocol}://${validated.host}:${validated.port}`;

    return validated;
  }

  /**
   * Validate AI configuration
   */
  validateAIConfig() {
    const aiConfig = config.getSection('ai');
    
    return {
      defaultModel: this.validateRequired('ai.default_model', aiConfig.default_model, 'llama3'),
      embeddingModel: this.validateRequired('ai.embedding_model', aiConfig.embedding_model, 'nomic-embed-text'),
      embeddingDimensions: this.validateNumber('ai.embedding_dimensions', aiConfig.embedding_dimensions, 768),
      maxTokens: this.validateNumber('ai.max_tokens', aiConfig.max_tokens, 4096),
      temperature: this.validateNumber('ai.temperature', aiConfig.temperature, 0.7, 0, 2),
      topP: this.validateNumber('ai.top_p', aiConfig.top_p, 0.9, 0, 1),
      topK: this.validateNumber('ai.top_k', aiConfig.top_k, 40),
      repeatPenalty: this.validateNumber('ai.repeat_penalty', aiConfig.repeat_penalty, 1.1)
    };
  }

  /**
   * Validate document processing configuration
   */
  validateDocumentProcessingConfig() {
    const docConfig = config.getSection('document_processing');
    
    return {
      autoCleanupFiles: this.validateBoolean('document_processing.auto_cleanup_files', docConfig.auto_cleanup_files, true),
      cleanupDelaySeconds: this.validateNumber('document_processing.cleanup_delay_seconds', docConfig.cleanup_delay_seconds, 30),
      keepFailedFiles: this.validateBoolean('document_processing.keep_failed_files', docConfig.keep_failed_files, true),
      cleanupLogLevel: this.validateRequired('document_processing.cleanup_log_level', docConfig.cleanup_log_level, 'info'),
      maxFileAgeDays: this.validateNumber('document_processing.max_file_age_days', docConfig.max_file_age_days, 7),
      trackCleanupStats: this.validateBoolean('document_processing.track_cleanup_stats', docConfig.track_cleanup_stats, true)
    };
  }

  /**
   * Validate image processing configuration
   */
  validateImageProcessingConfig() {
    const imgConfig = config.getSection('image_processing');
    
    return {
      enabled: this.validateBoolean('image_processing.enabled', imgConfig.enabled, true),
      dockerContainer: this.validateRequired('image_processing.docker_container', imgConfig.docker_container, 'productdemo-image-processor'),
      minSizeKb: this.validateNumber('image_processing.min_size_kb', imgConfig.min_size_kb, 5),
      minWidth: this.validateNumber('image_processing.min_width', imgConfig.min_width, 100),
      minHeight: this.validateNumber('image_processing.min_height', imgConfig.min_height, 100),
      maxImagesPerDocument: this.validateNumber('image_processing.max_images_per_document', imgConfig.max_images_per_document, 100),
      ocrEnabled: this.validateBoolean('image_processing.ocr_enabled', imgConfig.ocr_enabled, true),
      base64Encoding: this.validateBoolean('image_processing.base64_encoding', imgConfig.base64_encoding, true),
      maxImagesInResponse: this.validateNumber('image_processing.max_images_in_response', imgConfig.max_images_in_response, 3),
      similarityThreshold: this.validateNumber('image_processing.similarity_threshold', imgConfig.similarity_threshold, 0.7, 0, 1),
      keywordBoostFactor: this.validateNumber('image_processing.keyword_boost_factor', imgConfig.keyword_boost_factor, 1.2)
    };
  }

  /**
   * Validate MCP configuration
   */
  validateMCPConfig() {
    const mcpConfig = config.getSection('mcp-server');

    return {
      terminalInfoEndpoint: this.validateRequired('mcp-server.mcp_terminal_info_endpoint', mcpConfig.mcp_terminal_info_endpoint, '/info'),
      defaultHost: this.validateRequired('mcp-server.mcp_terminal_command_default_host_1', mcpConfig.mcp_terminal_command_default_host_1, 'localhost'),
      defaultTimeout: this.validateNumber('mcp-server.mcp_terminal_command_default_timeout_1', mcpConfig.mcp_terminal_command_default_timeout_1, 300)
    };
  }

  /**
   * Validate Chat2SQL configuration
   */
  validateChat2SqlConfig() {
    const chat2sqlConfig = config.getSection('chat2sql');

    const validated = {
      enabled: this.validateBoolean('chat2sql.enabled', chat2sqlConfig.enabled, true),
      protocol: this.validateRequired('chat2sql.protocol', chat2sqlConfig.protocol, 'http'),
      host: this.validateRequired('chat2sql.host', chat2sqlConfig.host, 'localhost'),
      port: this.validatePort('chat2sql.port', chat2sqlConfig.port, 5000),
      connectionTimeout: this.validateNumber('chat2sql.connection_timeout', chat2sqlConfig.connection_timeout, 30000),
      requestTimeout: this.validateNumber('chat2sql.request_timeout', chat2sqlConfig.request_timeout, 60000),
      dockerContainer: this.validateRequired('chat2sql.docker_container', chat2sqlConfig.docker_container, 'productdemo-chat2sql')
    };

    // Construct full URL
    validated.baseUrl = `${validated.protocol}://${validated.host}:${validated.port}`;

    return validated;
  }

  /**
   * Validate required field
   */
  validateRequired(key, value, defaultValue) {
    if (value === undefined || value === null || value === '') {
      if (defaultValue !== undefined) {
        this.warnings.push(`Using default value for ${key}: ${defaultValue}`);
        return defaultValue;
      } else {
        this.validationErrors.push(`Required configuration missing: ${key}`);
        return null;
      }
    }
    return value;
  }

  /**
   * Validate port number
   */
  validatePort(key, value, defaultValue) {
    const port = parseInt(value) || defaultValue;
    if (port < 1 || port > 65535) {
      this.validationErrors.push(`Invalid port number for ${key}: ${port}`);
      return defaultValue;
    }
    return port;
  }

  /**
   * Validate number
   */
  validateNumber(key, value, defaultValue, min = null, max = null) {
    const num = parseFloat(value);
    if (isNaN(num)) {
      if (defaultValue !== undefined) {
        this.warnings.push(`Invalid number for ${key}, using default: ${defaultValue}`);
        return defaultValue;
      } else {
        this.validationErrors.push(`Invalid number for ${key}: ${value}`);
        return null;
      }
    }
    
    if (min !== null && num < min) {
      this.validationErrors.push(`Value for ${key} below minimum (${min}): ${num}`);
      return defaultValue;
    }
    
    if (max !== null && num > max) {
      this.validationErrors.push(`Value for ${key} above maximum (${max}): ${num}`);
      return defaultValue;
    }
    
    return num;
  }

  /**
   * Validate boolean
   */
  validateBoolean(key, value, defaultValue) {
    if (typeof value === 'boolean') {
      return value;
    }
    
    if (typeof value === 'string') {
      const lower = value.toLowerCase();
      if (lower === 'true' || lower === '1' || lower === 'yes') {
        return true;
      }
      if (lower === 'false' || lower === '0' || lower === 'no') {
        return false;
      }
    }
    
    this.warnings.push(`Invalid boolean for ${key}, using default: ${defaultValue}`);
    return defaultValue;
  }

  /**
   * Get validated configuration
   */
  getValidatedConfig() {
    if (!this.validatedConfig) {
      throw new Error('Configuration not validated yet. Call validateConfiguration() first.');
    }
    return this.validatedConfig;
  }

  /**
   * Get all port mappings for documentation/debugging
   */
  getPortMappings() {
    const config = this.getValidatedConfig();
    
    return {
      'Main Application': {
        port: config.server.port,
        url: config.server.baseUrl,
        description: 'Backend + Frontend'
      },
      'ChromaDB': {
        port: config.docker.chromadb.port,
        url: config.docker.chromadb.url,
        description: 'Vector Database'
      },
      'Redis': {
        port: config.redis.port,
        url: `redis://${config.redis.host}:${config.redis.port}`,
        description: 'Queue Management'
      },
      'PostgreSQL': {
        port: config.database.port,
        url: `postgresql://${config.database.host}:${config.database.port}/${config.database.name}`,
        description: 'Primary Database'
      },
      'Ollama': {
        port: config.ollama.port,
        url: config.ollama.baseUrl,
        description: 'AI Model Inference'
      },
      'Chat2SQL': {
        port: config.chat2sql.port,
        url: config.chat2sql.baseUrl,
        description: 'Natural Language to SQL Service'
      }
    };
  }
}

module.exports = ConfigValidationService;
