import React, { useEffect, useState, useRef } from 'react';
import { getActiveOllamaModels, OllamaModel } from '../../services/ollamaService';
import {
  Box,
  Text,
  Flex,
  Tooltip,
  Badge,
  Spinner,
  Alert,
  AlertIcon,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverBody,
  PopoverArrow,
  PopoverCloseButton,
  Button,
  Icon
} from '@chakra-ui/react';
import { CpuChipIcon, ChevronDownIcon } from '@heroicons/react/24/outline';
import { animations, messageBubbleStyles } from './chatStyles'; // Fixed import path

interface ModelSelectorProps {
  onSelectModel: (modelId: string) => void;
  selectedModelId?: string;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  onSelectModel,
  selectedModelId
}) => {
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  // Load models from API
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const fetchedModels = await getActiveOllamaModels();
        setModels(fetchedModels);

        if (!selectedModelId && fetchedModels.length > 0) {
          onSelectModel(fetchedModels[0].id);
        }

        setError(null);
      } catch (err) {
        console.error('Failed to fetch models:', err);
        setError('Failed to load AI models. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [onSelectModel, selectedModelId]);

  // Handle model selection
  const handleModelChange = (modelId: string) => {
    onSelectModel(modelId);
    localStorage.setItem('selectedModelId', modelId);
    setIsOpen(false);
  };

  // Get model badge type
  const getModelBadgeType = (modelName: string) => {
    const lowerName = modelName.toLowerCase();
    if (lowerName.includes('code') || lowerName.includes('starcoder')) {
      return { color: 'purple', text: 'Code' };
    } else if (lowerName.includes('mini') || lowerName.includes('tiny')) {
      return { color: 'green', text: 'Fast' };
    } else if (lowerName.includes('large') || lowerName.includes('34b') || lowerName.includes('70b')) {
      return { color: 'orange', text: 'Large' };
    }
    return { color: 'blue', text: 'General' };
  };

  // Add animation styles
  useEffect(() => {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
      ${animations.fadeIn}
      ${animations.slideIn}
      .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12Frenchrgba(0,0,0,0.1);
      }
    `;
    document.head.appendChild(styleElement);

    return () => {
      document.head.removeChild(styleElement); // Cleanup without returning the element
    };
  }, []);

  if (loading) {
    return (
      <Flex align="center" justify="center" p={2}>
        <Spinner size="sm" mr={2} color="var(--color-primary)" />
        <Text fontSize="sm" color="var(--color-text-muted)">Loading models...</Text>
      </Flex>
    );
  }

  if (error) {
    return (
      <Alert status="error" variant="subtle" borderRadius="md">
        <AlertIcon />
        {error}
      </Alert>
    );
  }

  if (models.length === 0) {
    return (
      <Alert status="info" variant="subtle" borderRadius="md">
        <AlertIcon />
        No AI models available. Please ask an administrator to configure models.
      </Alert>
    );
  }

  const selectedModel = models.find(m => m.id === selectedModelId);

  return (
    <Box py={1} maxW="250px">
      <Popover isOpen={isOpen} onClose={() => setIsOpen(false)} placement="bottom-end">
        <PopoverTrigger>
          <Button
            onClick={() => setIsOpen(!isOpen)}
            variant="outline"
            size="sm"
            borderRadius="full"
            bg="transparent"
            borderColor="rgba(255, 255, 255, 0.15)"
            color="var(--color-text)"
            _hover={{ bg: 'rgba(255, 255, 255, 0.05)', transform: 'scale(1.02)' }}
            transition="all 0.2s ease"
            leftIcon={<Icon as={CpuChipIcon} w={3.5} h={3.5} />}
            rightIcon={<Icon as={ChevronDownIcon} w={3.5} h={3.5} />}
            px={3}
            py={1}
            animation="fadeIn 0.3s ease-in-out"
          >
            <Text fontSize="xs" fontWeight="medium" isTruncated maxW="120px">
              {selectedModel?.name || 'Select Model'}
            </Text>
            {selectedModel && (
              <Badge
                ml={1}
                fontSize="0.6em"
                colorScheme={getModelBadgeType(selectedModel.name).color}
                variant="subtle"
                borderRadius="full"
                px={1.5}
              >
                {getModelBadgeType(selectedModel.name).text}
              </Badge>
            )}
          </Button>
        </PopoverTrigger>

        <PopoverContent
          bg="rgba(var(--color-surface-light-rgb), 0.7)"
          borderColor="rgba(255, 255, 255, 0.15)"
          borderRadius="md"
          boxShadow="0 4px 12px rgba(0,0,0,0.1)"
          width="300px"
          ref={popoverRef}
          animation="slideIn 0.2s ease-out"
          sx={{
            backdropFilter: "blur(5px)",
            WebkitBackdropFilter: "blur(5px)"
          }}
        >
          <PopoverArrow bg="rgba(var(--color-surface-light-rgb), 0.7)" />
          <PopoverCloseButton color="var(--color-text-muted)" />
          <PopoverBody p={2}>
            {models.map(model => (
              <Tooltip
                key={model.id}
                label={model.description || 'No description available'}
                placement="right"
                hasArrow
                bg="var(--color-surface-dark)"
                color="var(--color-text)"
                borderRadius="md"
              >
                <Box
                  className="model-card"
                  p={3}
                  mb={1}
                  borderRadius="md"
                  bg={selectedModelId === model.id ? 'var(--color-primary-translucent)' : 'transparent'}
                  _hover={{ bg: 'var(--color-surface-dark)' }}
                  cursor="pointer"
                  onClick={() => handleModelChange(model.id)}
                  transition="all 0.2s ease"
                  display="flex"
                  alignItems="center"
                  justifyContent="space-between"
                >
                  <Box>
                    <Text fontSize="sm" fontWeight="medium" color="var(--color-text)">
                      {model.name}
                    </Text>
                    <Text fontSize="xs" color="var(--color-text-muted)" noOfLines={2}>
                      {model.description || 'AI model'}
                    </Text>
                  </Box>
                  <Badge
                    fontSize="0.7em"
                    colorScheme={getModelBadgeType(model.name).color}
                    variant="subtle"
                    borderRadius="full"
                  >
                    {getModelBadgeType(model.name).text}
                  </Badge>
                </Box>
              </Tooltip>
            ))}
          </PopoverBody>
        </PopoverContent>
      </Popover>
    </Box>
  );
};

export default ModelSelector;