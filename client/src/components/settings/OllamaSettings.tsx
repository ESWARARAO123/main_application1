import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Grid,
  Input,
  Text,
  useToast,
  Spinner,
  Flex,
  Switch,
  VStack,
  HStack,
  Badge,
  Tooltip,
  Heading,
  Alert,
  AlertIcon,
  AlertDescription,
  Divider,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Select,
  Tag,
  TagLabel,
  TagLeftIcon,
  Icon,
  keyframes,
  Kbd,
  useColorModeValue,
  IconButton,
  Checkbox,
} from '@chakra-ui/react';
import {
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer
} from '@chakra-ui/table';
import { Card, CardBody, CardHeader, CardFooter } from '@chakra-ui/card';
import {
  CheckCircleIcon,
  WarningIcon,
  RepeatIcon,
  RepeatClockIcon,
  SettingsIcon,
  LinkIcon,
  StarIcon,
  ArrowForwardIcon,
  CloseIcon,
  ChevronRightIcon,
  InfoIcon
} from '@chakra-ui/icons';
import axios from 'axios';
import { syncOllamaModels, SyncResult, getOllamaSettings, getOllamaModelsFromDB, saveOllamaSettings, testOllamaConnection, getAvailableOllamaModels, toggleOllamaModelStatus, OllamaSettings as ServiceOllamaSettings, OllamaModel } from '../../services/ollamaService';

// Storage key for caching settings
const SETTINGS_CACHE_KEY = 'ollama_settings_cache';
const MODELS_CACHE_KEY = 'ollama_models_cache';
const AVAILABLE_MODELS_CACHE_KEY = 'ollama_available_models_cache';
const SELECTED_MODELS_CACHE_KEY = 'ollama_selected_models_cache';

// Animation keyframes
const pulse = keyframes`
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
`;

interface OllamaSettingsProps {
  isAdmin: boolean;
}

// Use OllamaSettings type from the service
type Settings = ServiceOllamaSettings;

const OllamaSettings: React.FC<OllamaSettingsProps> = ({ isAdmin }) => {
  const [settings, setSettings] = useState<Settings>(() => {
    const cachedSettings = localStorage.getItem(SETTINGS_CACHE_KEY);
    return cachedSettings ? JSON.parse(cachedSettings) : {
      host: 'localhost',
      port: 11434,
      default_model: ''
    };
  });

  const [models, setModels] = useState<OllamaModel[]>(() => {
    const cachedModels = localStorage.getItem(MODELS_CACHE_KEY);
    return cachedModels ? JSON.parse(cachedModels) : [];
  });

  const [availableModels, setAvailableModels] = useState<OllamaModel[]>(() => {
    const cachedAvailableModels = localStorage.getItem(AVAILABLE_MODELS_CACHE_KEY);
    return cachedAvailableModels ? JSON.parse(cachedAvailableModels) : [];
  });

  const [selectedModels, setSelectedModels] = useState<string[]>(() => {
    const cachedSelectedModels = localStorage.getItem(SELECTED_MODELS_CACHE_KEY);
    return cachedSelectedModels ? JSON.parse(cachedSelectedModels) : [];
  });

  const [saveLoading, setSaveLoading] = useState(false);
  const [testLoading, setTestLoading] = useState(false);
  const [fetchLoading, setFetchLoading] = useState(false);
  const [syncLoading, setSyncLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'unknown' | 'success' | 'error' | 'testing'>('unknown');
  const [lastAction, setLastAction] = useState<string>('');
  const [lastError, setLastError] = useState<string>('');
  const toast = useToast();
  const retryModal = useDisclosure();
  const [retryOperation, setRetryOperation] = useState<() => Promise<void>>(() => async () => {});

  useEffect(() => {
    if (isAdmin) {
      loadSettings();
      loadModels();
    }
  }, [isAdmin]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_CACHE_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    localStorage.setItem(MODELS_CACHE_KEY, JSON.stringify(models));
  }, [models]);

  useEffect(() => {
    localStorage.setItem(AVAILABLE_MODELS_CACHE_KEY, JSON.stringify(availableModels));
  }, [availableModels]);

  useEffect(() => {
    localStorage.setItem(SELECTED_MODELS_CACHE_KEY, JSON.stringify(selectedModels));
  }, [selectedModels]);

  const loadSettings = useCallback(async () => {
    try {
      setLastAction('Loading settings...');
      const result = await getOllamaSettings();

      if (result && typeof result === 'object') {
        setSettings(result);
        setLastAction('Settings loaded successfully');
      } else {
        setLastAction('Failed to load settings: Invalid response format');
        setLastError('Could not load Ollama settings');
        showToast({
          title: 'Error loading settings',
          description: 'Could not load Ollama settings',
          status: 'error'
        });
      }
    } catch (error: any) {
      console.error('Error loading Ollama settings:', error);
      setLastAction('Failed to load settings');
      setLastError(error.response?.data?.message || error.message || 'Failed to load Ollama settings');

      showToast({
        title: 'Error loading settings',
        description: error.response?.data?.message || error.message || 'Failed to load Ollama settings',
        status: 'error'
      });

      setRetryOperation(() => loadSettings);
    }
  }, []);

  const loadModels = useCallback(async () => {
    try {
      setModelsLoading(true);
      setLastAction('Loading models from database...');

      const models = await getOllamaModelsFromDB();

      if (Array.isArray(models)) {
        setModels(models || []);
        setLastAction(`Loaded ${models.length || 0} models from database`);
      } else {
        setLastAction('Failed to load models: Invalid response format');
        setLastError('Could not load AI models from database');
        showToast({
          title: 'Error loading models',
          description: 'Could not load AI models from database',
          status: 'error'
        });

        setRetryOperation(() => loadModels);
      }
    } catch (error: any) {
      console.error('Error loading AI models:', error);
      setLastAction('Failed to load models from database');
      setLastError(error.response?.data?.message || error.message || 'Failed to load AI models from database');

      showToast({
        title: 'Error loading models',
        description: error.response?.data?.message || error.message || 'Failed to load AI models from database',
        status: 'error'
      });

      setRetryOperation(() => loadModels);
    } finally {
      setModelsLoading(false);
    }
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSettings(prev => ({
      ...prev,
      [name]: name === 'port' ? parseInt(value, 10) || 0 : value
    }));
  };

  const saveSettings = async () => {
    try {
      setSaveLoading(true);
      setLastAction('Saving settings...');
      setLastError('');

      const result = await saveOllamaSettings(settings);

      if (result && typeof result === 'object') {
        setLastAction('Settings saved successfully');
        setConnectionStatus('unknown');
        showToast({
          title: 'Settings saved',
          description: 'Ollama connection settings have been updated',
          status: 'success'
        });

        localStorage.setItem(SETTINGS_CACHE_KEY, JSON.stringify(settings));

        await testConnection();
      } else {
        setLastAction('Failed to save settings: Invalid response format');
        setLastError('There was a problem saving your settings');
        showToast({
          title: 'Error saving settings',
          description: 'There was a problem saving your settings',
          status: 'error'
        });

        setRetryOperation(() => saveSettings);
        retryModal.onOpen();
      }
    } catch (error: any) {
      console.error('Error saving Ollama settings:', error);
      setLastAction('Failed to save settings');
      setLastError(error.response?.data?.message || error.message || 'Failed to save Ollama connection settings');

      showToast({
        title: 'Error saving settings',
        description: error.response?.data?.message || error.message || 'Failed to save Ollama connection settings',
        status: 'error'
      });

      setRetryOperation(() => saveSettings);
      retryModal.onOpen();
    } finally {
      setSaveLoading(false);
    }
  };

  const testConnection = async () => {
    try {
      setTestLoading(true);
      setConnectionStatus('testing');
      setLastAction('Testing connection to Ollama server...');
      setLastError('');

      const result = await testOllamaConnection(settings.host, settings.port);

      if (result.success) {
        setConnectionStatus('success');
        setLastAction(`Connection successful! Ollama ${result.version || 'server'} responded in ${result.responseTime || '?'}ms`);
        showToast({
          title: 'Connection successful',
          description: `Connected to Ollama ${result.version || 'server'} in ${result.responseTime || '?'}ms`,
          status: 'success'
        });

        fetchAvailableModels();
      } else {
        setConnectionStatus('error');
        setLastAction(`Connection failed: ${result.message || result.error || 'Unknown error'}`);
        setLastError(result.message || result.error || 'Failed to connect to Ollama server');
        showToast({
          title: 'Connection failed',
          description: result.message || result.error || 'Failed to connect to Ollama server',
          status: 'error'
        });
      }
    } catch (error: any) {
      console.error('Error testing Ollama connection:', error);
      setConnectionStatus('error');
      setLastAction('Connection test failed');
      setLastError(error.response?.data?.message || error.message || 'Failed to connect to Ollama server');

      showToast({
        title: 'Connection failed',
        description: error.response?.data?.message || error.message || 'Failed to connect to Ollama server',
        status: 'error'
      });
    } finally {
      setTestLoading(false);
    }
  };

  const showToast = ({ title, description, status }: { title: string, description: string, status: 'success' | 'error' | 'info' | 'warning' }) => {
    toast({
      title,
      description,
      status,
      duration: 3000,
      isClosable: true,
      position: 'top-right',
      containerStyle: {
        marginTop: '1rem',
      },
    });
  };

  const handleModelSelection = (modelId: string) => {
    setSelectedModels(prev => {
      if (prev.includes(modelId)) {
        return prev.filter(id => id !== modelId);
      } else {
        return [...prev, modelId];
      }
    });
  };

  const fetchAvailableModels = async () => {
    try {
      setFetchLoading(true);
      setLastAction('Fetching available models from Ollama server...');
      setLastError('');

      const models = await getAvailableOllamaModels();

      if (Array.isArray(models)) {
        setAvailableModels(models);

        if (selectedModels.length === 0) {
          setSelectedModels(models.map(model => model.ollama_model_id || model.name));
        }

        setLastAction(`Found ${models.length} models on Ollama server`);
        showToast({
          title: 'Models fetched',
          description: `Found ${models.length} models on the Ollama server`,
          status: 'success'
        });
      } else {
        setLastAction('Failed to fetch models: Invalid response format');
        setLastError('Could not fetch models from Ollama server');
        showToast({
          title: 'Fetch failed',
          description: 'Could not fetch models from Ollama server',
          status: 'error'
        });

        setRetryOperation(() => fetchAvailableModels);
        retryModal.onOpen();
      }
    } catch (error: any) {
      console.error('Error fetching available models:', error);
      setLastAction('Failed to fetch models from Ollama server');
      setLastError(error.response?.data?.message || error.message || 'Error fetching models from Ollama server');

      showToast({
        title: 'Fetch failed',
        description: error.response?.data?.message || error.message || 'Error fetching models from Ollama server',
        status: 'error'
      });

      setRetryOperation(() => fetchAvailableModels);
      retryModal.onOpen();
    } finally {
      setFetchLoading(false);
    }
  };

  const formatModelSize = (sizeInBytes: number): string => {
    if (!sizeInBytes) return 'Unknown';

    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(sizeInBytes) / Math.log(1024));
    return `${(sizeInBytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
  };

  const syncModels = async () => {
    try {
      setSyncLoading(true);
      setLastAction('Saving model selections to database...');
      setLastError('');

      const result = await syncOllamaModels(selectedModels);

      if (result && typeof result === 'object') {
        const totalSynced = (result.added || 0) + (result.updated || 0);
        const inactivatedCount = result.inactivated || 0;

        setLastAction(`Models saved successfully: ${totalSynced} models updated (${result.added || 0} added, ${result.updated || 0} updated, ${inactivatedCount} deactivated)`);

        showToast({
          title: 'Models saved',
          description: `Successfully saved ${selectedModels.length} models as active and deactivated ${inactivatedCount} models`,
          status: 'success'
        });

        await loadModels();
      } else {
        setLastAction('Failed to save models: Invalid response format');
        setLastError('Failed to save models to database - unexpected response format');

        showToast({
          title: 'Save failed',
          description: 'Failed to save models to database - unexpected response format',
          status: 'error'
        });

        setRetryOperation(() => syncModels);
        retryModal.onOpen();
      }
    } catch (error: any) {
      console.error('Error saving models:', error);
      setLastAction('Failed to save models to database');
      setLastError(error.response?.data?.message || error.message || 'Error saving models to database');

      showToast({
        title: 'Save failed',
        description: error.response?.data?.message || error.message || 'Error saving models to database',
        status: 'error'
      });

      setRetryOperation(() => syncModels);
      retryModal.onOpen();
    } finally {
      setSyncLoading(false);
    }
  };

  const toggleModelStatus = async (modelId: string, isActive: boolean) => {
    try {
      setLastAction(`${isActive ? 'Activating' : 'Deactivating'} model ${modelId}...`);
      setLastError('');

      const model = await toggleOllamaModelStatus(modelId, isActive);

      if (model && typeof model === 'object') {
        setLastAction(`Model ${modelId} ${isActive ? 'activated' : 'deactivated'} successfully`);
        showToast({
          title: `Model ${isActive ? 'activated' : 'deactivated'}`,
          description: `The model has been ${isActive ? 'activated' : 'deactivated'} successfully`,
          status: 'success'
        });

        const updatedModels = models.map(m =>
          m.id === modelId ? { ...m, is_active: isActive } : m
        );
        setModels(updatedModels);

        localStorage.setItem(MODELS_CACHE_KEY, JSON.stringify(updatedModels));
      } else {
        setLastAction(`Failed to ${isActive ? 'activate' : 'deactivate'} model: Invalid response format`);
        setLastError(`Could not ${isActive ? 'activate' : 'deactivate'} the model`);
        showToast({
          title: 'Status update failed',
          description: `Could not ${isActive ? 'activate' : 'deactivate'} the model`,
          status: 'error'
        });

        setRetryOperation(() => () => toggleModelStatus(modelId, isActive));
        retryModal.onOpen();
      }
    } catch (error: any) {
      console.error(`Error ${isActive ? 'activating' : 'deactivating'} model:`, error);
      setLastAction(`Failed to ${isActive ? 'activate' : 'deactivate'} model`);
      setLastError(error.response?.data?.message || error.message || `Error ${isActive ? 'activating' : 'deactivating'} the model`);

      showToast({
        title: 'Status update failed',
        description: error.response?.data?.message || error.message || `Error ${isActive ? 'activating' : 'deactivating'} the model`,
        status: 'error'
      });

      setRetryOperation(() => () => toggleModelStatus(modelId, isActive));
      retryModal.onOpen();
    }
  };

  const handleRetry = async () => {
    setIsRetrying(true);
    try {
      await retryOperation();
      retryModal.onClose();
    } catch (error) {
      console.error('Retry failed:', error);
    } finally {
      setIsRetrying(false);
    }
  };

  const renderConnectionStatusIcon = () => {
    if (connectionStatus === 'success') {
      return <CheckCircleIcon color="green.400" />;
    } else if (connectionStatus === 'error') {
      return <WarningIcon color="red.400" />;
    } else if (connectionStatus === 'testing') {
      return <Spinner size="sm" color="blue.400" />;
    } else {
      return <RepeatIcon color="gray.400" />;
    }
  };

  const handleDefaultModelChange = (modelId: string) => {
    setSettings(prev => ({
      ...prev,
      default_model: modelId
    }));
  };

  if (!isAdmin) {
    return (
      <Box px={4} py={4} width="100%">
        <Alert
          status="warning"
          borderRadius="2xl"
          boxShadow="md"
          p={4}
          bgGradient="linear(to-r, orange.50, orange.100)"
          transition="all 0.3s ease"
        >
          <AlertIcon />
          <AlertDescription fontWeight="medium">
            You need administrator privileges to access Ollama settings.
          </AlertDescription>
        </Alert>
      </Box>
    );
  }

  return (
    <Box width="100%" px={4} py={4}>
      <VStack gap={8} align="stretch">
        <Card
          variant="elevated"
          boxShadow="md"
          borderRadius="xl"
          overflow="hidden"
          borderWidth="1px"
          borderColor="border.base"
          bgGradient="linear(to-b, surface.base, surface.dark)"
          transition="all 0.3s ease"
          _hover={{
            boxShadow: "lg",
            transform: "translateY(-2px)"
          }}
        >
          <CardHeader
            bgGradient="linear(to-r, surface.light, surface.base)"
            py={3}
            px={5}
            borderBottomWidth="1px"
            borderColor="border.base"
            display="flex"
            alignItems="center"
            borderTopRadius="xl"
          >
            <Icon as={SettingsIcon} mr={2} color="brand.primary" boxSize={5} />
            <Heading size="md" color="text.base" fontWeight="semibold">
              AI API Connection
            </Heading>
          </CardHeader>

          <CardBody p={5} bg="transparent">
            <Grid templateColumns={{ base: "1fr", md: "repeat(2, 1fr)" }} gap={6}>
              <FormControl>
                <FormLabel
                  fontWeight="semibold"
                  color="text.secondary"
                  display="flex"
                  alignItems="center"
                  fontSize="sm"
                >
                  <Icon as={LinkIcon} mr={2} boxSize={4} color="brand.primary" />
                  Host
                </FormLabel>
                <Input
                  name="host"
                  value={settings.host}
                  onChange={handleInputChange}
                  placeholder="localhost"
                  bg="surface.dark"
                  borderColor="border.base"
                  borderRadius="2xl"
                  _hover={{
                    borderColor: "brand.primary",
                    boxShadow: "0 0 0 1px rgba(59, 130, 246, 0.8)"
                  }}
                  _focus={{
                    borderColor: "brand.primary",
                    boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.3)"
                  }}
                  color="text.base"
                  size="lg"
                  fontFamily="mono"
                  px={5}
                  py={6}
                  transition="all 0.3s ease"
                />
              </FormControl>

              <FormControl>
                <FormLabel
                  fontWeight="semibold"
                  color="text.secondary"
                  display="flex"
                  alignItems="center"
                  fontSize="sm"
                >
                  <Icon as={RepeatIcon} mr={2} boxSize={4} color="brand.primary" />
                  Port
                </FormLabel>
                <Input
                  name="port"
                  type="number"
                  value={settings.port}
                  onChange={handleInputChange}
                  placeholder="11434"
                  bg="surface.dark"
                  borderColor="border.base"
                  borderRadius="2xl"
                  _hover={{
                    borderColor: "brand.primary",
                    boxShadow: "0 0 0 1px rgba(59, 130, 246, 0.8)"
                  }}
                  _focus={{
                    borderColor: "brand.primary",
                    boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.3)"
                  }}
                  color="text.base"
                  size="lg"
                  fontFamily="mono"
                  px={5}
                  py={6}
                  transition="all 0.3s ease"
                />
              </FormControl>

              <FormControl gridColumn={{ md: "span 2" }}>
                <FormLabel
                  fontWeight="semibold"
                  color="text.secondary"
                  display="flex"
                  alignItems="center"
                  fontSize="xs"
                  mb={1}
                >
                  <Icon as={StarIcon} mr={1} boxSize={3} color="brand.primary" />
                  Default Model
                </FormLabel>
                {models.length > 0 ? (
                  <Select
                    name="default_model"
                    value={settings.default_model}
                    onChange={(e) => handleDefaultModelChange(e.target.value)}
                    placeholder="Select a default model"
                    bg="surface.dark"
                    borderColor="border.base"
                    borderRadius="md"
                    _hover={{
                      borderColor: "brand.primary",
                      boxShadow: "0 0 0 1px rgba(59, 130, 246, 0.8)"
                    }}
                    _focus={{
                      borderColor: "brand.primary",
                      boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.3)"
                    }}
                    color="text.base"
                    size="md"
                    icon={<ChevronRightIcon />}
                    px={3}
                    py={1}
                    transition="all 0.3s ease"
                    sx={{
                      '& option': {
                        background: 'var(--color-surface-dark)',
                        color: 'var(--color-text)',
                        padding: '6px',
                        margin: '2px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                      },
                      '&::-webkit-scrollbar': {
                        width: '8px',
                      },
                      '&::-webkit-scrollbar-track': {
                        background: 'rgba(21, 28, 44, 0.5)',
                        borderRadius: '4px'
                      },
                      '&::-webkit-scrollbar-thumb': {
                        background: 'rgba(59, 130, 246, 0.7)',
                        borderRadius: '4px',
                        '&:hover': {
                          background: 'rgba(59, 130, 246, 0.9)',
                        }
                      },
                    }}
                  >
                    {models.filter(model => model.is_active).map(model => (
                      <option key={model.ollama_model_id} value={model.ollama_model_id}>
                        {model.name}
                      </option>
                    ))}
                  </Select>
                ) : (
                  <Flex direction="column">
                    <Input
                      name="default_model"
                      value={settings.default_model}
                      onChange={handleInputChange}
                      placeholder="Sync models first to select a default model"
                      bg="surface.dark"
                      borderColor="border.base"
                      borderRadius="md"
                      _hover={{
                        borderColor: "brand.primary",
                        boxShadow: "0 0 0 1px rgba(59, 130, 246, 0.8)"
                      }}
                      _focus={{
                        borderColor: "brand.primary",
                        boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.3)"
                      }}
                      color="text.base"
                      isDisabled={true}
                      size="md"
                      px={3}
                      py={2}
                      transition="all 0.3s ease"
                    />
                    <Text mt={3} fontSize="sm" color="text.muted" display="flex" alignItems="center">
                      <Box
                        as="span"
                        display="inline-block"
                        w="8px"
                        h="8px"
                        borderRadius="full"
                        bg="orange.400"
                        mr={2}
                        animation={`${pulse} 2s infinite`}
                      />
                      Sync models to enable selection
                    </Text>
                  </Flex>
                )}
              </FormControl>
            </Grid>

            <Flex
              direction="column"
              gap={6}
              mt={10}
              borderTop="1px solid"
              borderColor="border.base"
              pt={8}
            >
              <HStack
                gap={4}
                flexWrap="wrap"
                justifyContent="space-between"
              >
                <Button
                  bgGradient="linear(to-r, brand.primary, brand.primary-dark)"
                  color="white"
                  onClick={saveSettings}
                  isLoading={saveLoading}
                  loadingText="Saving"
                  leftIcon={saveLoading ? undefined : <CheckCircleIcon />}
                  _hover={{
                    bgGradient: "linear(to-r, brand.primary-dark, brand.primary)",
                    transform: "translateY(-1px)",
                    boxShadow: "md"
                  }}
                  _active={{
                    transform: "translateY(0)",
                    boxShadow: "sm"
                  }}
                  size="md"
                  borderRadius="md"
                  px={4}
                  py={2}
                  fontWeight="medium"
                  transition="all 0.3s ease"
                  boxShadow="sm"
                >
                  Save Settings
                </Button>

                <Button
                  bgGradient={
                    connectionStatus === 'success' ? 'linear(to-r, green.500, green.600)' :
                    connectionStatus === 'error' ? 'linear(to-r, red.500, red.600)' :
                    connectionStatus === 'testing' ? 'linear(to-r, blue.500, blue.600)' :
                    'linear(to-r, gray.500, gray.600)'
                  }
                  color="white"
                  onClick={testConnection}
                  isLoading={testLoading}
                  loadingText="Testing"
                  leftIcon={renderConnectionStatusIcon()}
                  _hover={{
                    bgGradient:
                      connectionStatus === 'success' ? 'linear(to-r, green.600, green.700)' :
                      connectionStatus === 'error' ? 'linear(to-r, red.600, red.700)' :
                      connectionStatus === 'testing' ? 'linear(to-r, blue.600, blue.700)' :
                      'linear(to-r, gray.600, gray.700)',
                    transform: "translateY(-1px)",
                    boxShadow: "md"
                  }}
                  _active={{
                    transform: "translateY(0)",
                    boxShadow: "sm"
                  }}
                  size="md"
                  borderRadius="md"
                  px={4}
                  py={2}
                  fontWeight="medium"
                  transition="all 0.3s ease"
                  boxShadow="sm"
                >
                  {connectionStatus === 'success' ? 'Connection Successful' :
                   connectionStatus === 'error' ? 'Connection Failed' :
                   connectionStatus === 'testing' ? 'Testing Connection...' : 'Test Connection'}
                </Button>
              </HStack>
            </Flex>
          </CardBody>
        </Card>

        <Card
          variant="elevated"
          boxShadow="md"
          borderRadius="xl"
          overflow="hidden"
          borderWidth="1px"
          borderColor="border.base"
          bgGradient="linear(to-b, surface.base, surface.dark)"
          transition="all 0.3s ease"
          _hover={{
            boxShadow: "lg",
            transform: "translateY(-2px)"
          }}
          mt={6}
        >
          <CardHeader
            bgGradient="linear(to-r, surface.light, surface.base)"
            py={3}
            px={5}
            borderBottomWidth="1px"
            borderColor="border.base"
            display="flex"
            alignItems="center"
            borderTopRadius="xl"
          >
            <Icon as={RepeatIcon} mr={2} color="brand.primary" boxSize={5} />
            <Heading size="md" color="text.base" fontWeight="semibold">
              Model Management
            </Heading>
          </CardHeader>

          <CardBody p={5} bg="transparent">
            <Box
              p={6}
              bg="surface.dark"
              borderRadius="2xl"
              borderWidth="1px"
              borderColor="border.base"
              mb={6}
              transition="all 0.3s ease"
              _hover={{
                bg: "surface.light",
                borderColor: "brand.primary"
              }}
            >
              <Flex justify="space-between" align="center" mb={4}>
                <Flex align="center">
                  <Box
                    as="span"
                    bgGradient="linear(to-r, brand.primary, brand.primary-dark)"
                    color="white"
                    fontWeight="bold"
                    w="32px"
                    h="32px"
                    borderRadius="full"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    mr={4}
                    fontSize="md"
                    boxShadow="md"
                  >
                    1
                  </Box>
                  <Text fontWeight="semibold" color="text.base" fontSize="lg">
                    Fetch Available Models
                  </Text>
                </Flex>

                <Button
                  bgGradient="linear(to-r, brand.primary, brand.primary-dark)"
                  color="white"
                  onClick={fetchAvailableModels}
                  isLoading={fetchLoading}
                  loadingText="Fetching"
                  leftIcon={<RepeatIcon />}
                  _hover={{
                    bgGradient: "linear(to-r, brand.primary-dark, brand.primary)",
                    transform: "translateY(-2px)",
                    boxShadow: "lg"
                  }}
                  _active={{
                    transform: "translateY(0)",
                    boxShadow: "md"
                  }}
                  size="lg"
                  borderRadius="full"
                  px={8}
                  py={6}
                  fontWeight="semibold"
                  transition="all 0.3s ease"
                  boxShadow="md"
                >
                  Fetch Models
                </Button>
              </Flex>

              <Text fontSize="sm" color="text.muted" ml={12}>
                Retrieve all available models from your Ollama server
              </Text>
            </Box>

            {availableModels.length > 0 && (
              <Box
                p={6}
                bg="surface.dark"
                borderRadius="2xl"
                borderWidth="1px"
                borderColor="border.base"
                mb={6}
                transition="all 0.3s ease"
                _hover={{
                  bg: "surface.light",
                  borderColor: "brand.primary"
                }}
              >
                <Flex justify="space-between" align="center" mb={4}>
                  <Flex align="center">
                    <Box
                      as="span"
                      bgGradient="linear(to-r, brand.primary, brand.primary-dark)"
                      color="white"
                      fontWeight="bold"
                      w="32px"
                      h="32px"
                      borderRadius="full"
                      display="flex"
                      alignItems="center"
                      justifyContent="center"
                      mr={4}
                      fontSize="md"
                      boxShadow="md"
                    >
                      2
                    </Box>
                    <Text fontWeight="semibold" color="text.base" fontSize="lg">
                      Select Models ({selectedModels.length} of {availableModels.length} selected)
                    </Text>
                  </Flex>

                  <Button
                    size="md"
                    onClick={() => setSelectedModels(availableModels.map(m => m.ollama_model_id || m.name))}
                    variant="outline"
                    borderRadius="full"
                    borderColor="brand.primary"
                    color="brand.primary"
                    _hover={{
                      bg: "rgba(59, 130, 246, 0.1)",
                      transform: "translateY(-1px)"
                    }}
                    transition="all 0.3s ease"
                  >
                    Select All
                  </Button>
                </Flex>

                <Text fontSize="sm" color="text.muted" ml={12} mb={6}>
                  Choose which models to make available in your application
                </Text>

                <VStack
                  spacing={3}
                  align="stretch"
                  maxH={{ base: "400px", md: "500px" }}
                  overflowY="auto"
                  ml={12}
                  pr={2}
                  css={{
                    '&::-webkit-scrollbar': {
                      width: '8px',
                    },
                    '&::-webkit-scrollbar-track': {
                      background: 'rgba(21, 28, 44, 0.5)',
                      borderRadius: '4px'
                    },
                    '&::-webkit-scrollbar-thumb': {
                      background: 'rgba(59, 130, 246, 0.7)',
                      borderRadius: '4px',
                      '&:hover': {
                        background: 'rgba(59, 130, 246, 0.9)',
                      }
                    },
                  }}
                >
                  {availableModels.map((model: any, index: number) => (
                    <Flex
                      key={index}
                      p={4}
                      borderWidth="1px"
                      borderRadius="xl"
                      borderColor={selectedModels.includes(model.ollama_model_id || model.name) ? "brand.primary" : "border.base"}
                      bg={selectedModels.includes(model.ollama_model_id || model.name) ? "surface.light" : "surface.dark"}
                      transition="all 0.3s ease"
                      cursor="pointer"
                      onClick={() => handleModelSelection(model.ollama_model_id || model.name)}
                      align="center"
                      justify="space-between"
                      _hover={{
                        transform: "translateY(-2px)",
                        boxShadow: "md",
                        bg: "surface.light"
                      }}
                      width="100%"
                    >
                      <Flex align="center" flex="1">
                        <Checkbox
                          isChecked={selectedModels.includes(model.ollama_model_id || model.name)}
                          onChange={(e) => {
                            e.stopPropagation();
                            handleModelSelection(model.ollama_model_id || model.name);
                          }}
                          colorScheme="green"
                          size="lg"
                          mr={4}
                        />

                        <Box>
                          <Text
                            fontSize="md"
                            fontWeight="semibold"
                            color="text.base"
                          >
                            {model.name}
                          </Text>
                        </Box>
                      </Flex>

                      <Flex gap={2} flexWrap="wrap" justify="flex-end">
                        {model.size && (
                          <Tag
                            size="sm"
                            colorScheme="blue"
                            borderRadius="full"
                            variant="subtle"
                            px={3}
                          >
                            {formatModelSize(model.size)}
                          </Tag>
                        )}

                        {model.name.includes('code') && (
                          <Tag
                            size="sm"
                            colorScheme="purple"
                            variant="subtle"
                            borderRadius="full"
                            px={3}
                          >
                            Code
                          </Tag>
                        )}

                        {model.name.includes('embed') && (
                          <Tag
                            size="sm"
                            colorScheme="orange"
                            variant="subtle"
                            borderRadius="full"
                            px={3}
                          >
                            Embed
                          </Tag>
                        )}
                      </Flex>
                    </Flex>
                  ))}
                </VStack>
              </Box>
            )}

            {availableModels.length > 0 && (
              <Box
                p={6}
                bg="surface.dark"
                borderRadius="2xl"
                borderWidth="1px"
                borderColor="border.base"
                transition="all 0.3s ease"
                _hover={{
                  bg: "surface.light",
                  borderColor: "brand.primary"
                }}
              >
                <Flex justify="space-between" align="center" mb={4}>
                  <Flex align="center">
                    <Box
                      as="span"
                      bgGradient="linear(to-r, brand.primary, brand.primary-dark)"
                      color="white"
                      fontWeight="bold"
                      w="32px"
                      h="32px"
                      borderRadius="full"
                      display="flex"
                      alignItems="center"
                      justifyContent="center"
                      mr={4}
                      fontSize="md"
                      boxShadow="md"
                    >
                      3
                    </Box>
                    <Text fontWeight="semibold" color="text.base" fontSize="lg">
                      Save Selected Models to Database
                    </Text>
                  </Flex>

                  <Button
                    bgGradient="linear(to-r, blue.500, blue.600)"
                    color="white"
                    isDisabled={selectedModels.length === 0}
                    onClick={syncModels}
                    isLoading={syncLoading}
                    loadingText="Saving"
                    size="lg"
                    borderRadius="full"
                    px={8}
                    py={6}
                    fontWeight="semibold"
                    transition="all 0.3s ease"
                    boxShadow="md"
                    _hover={{
                      bgGradient: "linear(to-r, blue.600, blue.700)",
                      transform: "translateY(-2px)",
                      boxShadow: "lg"
                    }}
                    _active={{
                      transform: "translateY(0)",
                      boxShadow: "md"
                    }}
                    _disabled={{
                      bgGradient: "linear(to-r, gray.500, gray.600)",
                      cursor: "not-allowed",
                      opacity: 0.7
                    }}
                  >
                    Save {selectedModels.length} Models
                  </Button>
                </Flex>

                <Text fontSize="sm" color="text.muted" ml={12}>
                  Only the selected models will be available in your application. Unselected models will be deactivated.
                </Text>
              </Box>
            )}

            {models.length > 0 && (
              <Box
                p={6}
                bgGradient="linear(to-r, surface.light, surface.dark)"
                borderRadius="2xl"
                borderWidth="1px"
                borderColor="border.base"
                mt={6}
                transition="all 0.3s ease"
                _hover={{
                  bgGradient: "linear(to-r, surface.base, surface.light)",
                  borderColor: "brand.primary"
                }}
              >
                <Flex align="center" mb={3}>
                  <CheckCircleIcon color="green.500" mr={3} boxSize={5} />
                  <Text fontWeight="semibold" color="text.base" fontSize="lg">
                    Database Status
                  </Text>
                </Flex>
                <Text fontSize="sm" color="text.muted" ml={8}>
                  <b>{models.filter(m => m.is_active).length}</b> active models in database out of <b>{models.length}</b> total
                  {models.filter(m => !m.is_active).length > 0 && (
                    <> • <b>{models.filter(m => !m.is_active).length}</b> inactive models</>
                  )}
                </Text>
              </Box>
            )}

            {lastAction && (
              <Box mt={6}>
                <Text fontSize="sm" color="text.muted" display="flex" alignItems="center">
                  <Box
                    as="span"
                    display="inline-block"
                    w="8px"
                    h="8px"
                    borderRadius="full"
                    bg="brand.primary"
                    mr={2}
                    animation={`${pulse} 2s infinite`}
                  />
                  <b>Last action:</b>&nbsp;{lastAction}
                </Text>
              </Box>
            )}
          </CardBody>
        </Card>
      </VStack>

      <Modal isOpen={retryModal.isOpen} onClose={retryModal.onClose}>
        <ModalOverlay
          bg="blackAlpha.800"
          backdropFilter="blur(8px)"
        />
        <ModalContent
          bg="surface.base"
          borderColor="border.base"
          borderWidth="1px"
          borderRadius="3xl"
          boxShadow="2xl"
          maxW="32rem"
        >
          <ModalHeader
            color="text.base"
            borderBottomWidth="1px"
            borderColor="border.base"
            display="flex"
            alignItems="center"
            fontSize="lg"
            fontWeight="semibold"
          >
            <WarningIcon color="red.500" mr={3} boxSize={5} />
            Operation Failed
          </ModalHeader>
          <ModalCloseButton
            color="text.base"
            borderRadius="full"
            _hover={{ bg: "surface.light" }}
            top={3}
            right={3}
          />

          <ModalBody py={8}>
            <Alert
              status="error"
              mb={6}
              borderRadius="xl"
              variant="left-accent"
              borderWidth="1px"
              borderColor="red.200"
              bgGradient="linear(to-r, red.50, red.100)"
              p={4}
            >
              <AlertIcon />
              <AlertDescription fontWeight="medium">{lastError}</AlertDescription>
            </Alert>

            <Text
              color="text.secondary"
              fontSize="md"
              display="flex"
              alignItems="center"
            >
              <Icon as={InfoIcon} mr={3} boxSize={4} color="blue.400" />
              Would you like to retry the operation?
            </Text>
          </ModalBody>

          <ModalFooter
            borderTopWidth="1px"
            borderColor="border.base"
            gap={4}
          >
            <Button
              variant="outline"
              onClick={retryModal.onClose}
              borderColor="border.base"
              color="text.base"
              borderRadius="full"
              px={6}
              py={5}
              _hover={{
                bg: 'surface.light',
                transform: "translateY(-1px)"
              }}
              transition="all 0.3s ease"
            >
              Cancel
            </Button>
            <Button
              bgGradient="linear(to-r, brand.primary, brand.primary-dark)"
              color="white"
              onClick={handleRetry}
              isLoading={isRetrying}
              loadingText="Retrying"
              leftIcon={<RepeatClockIcon />}
              borderRadius="full"
              px={6}
              py={5}
              fontWeight="semibold"
              _hover={{
                bgGradient: "linear(to-r, brand.primary-dark, brand.primary)",
                transform: "translateY(-2px)",
                boxShadow: "lg"
              }}
              _active={{
                transform: "translateY(0)",
                boxShadow: "md"
              }}
              transition="all 0.3s ease"
              boxShadow="md"
            >
              Retry
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default OllamaSettings;