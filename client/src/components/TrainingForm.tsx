import React, { useState } from 'react';
import { Box, Heading, FormControl, FormLabel, Input, Button, Text } from '@chakra-ui/react';

interface TrainingFormProps {
  onTrainingComplete?: (result: any) => void;
}

const TrainingForm: React.FC<TrainingFormProps> = ({ onTrainingComplete }) => {
  const [placeTable, setPlaceTable] = useState('ariane_place_sorted_csv');
  const [ctsTable, setCtsTable] = useState('ariane_cts_sorted_csv');
  const [routeTable, setRouteTable] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:8088/slack-prediction/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          place_table: placeTable,
          cts_table: ctsTable,
          ...(routeTable && { route_table: routeTable }),
        }),
      });

      const result = await response.json();

      if (!response.ok || result.status === 'error') {
        throw new Error(result.message || 'Failed to trigger training');
      }

      if (!result.place_to_cts) {
        throw new Error('Invalid response structure from server');
      }

      let trainingMessage = `
**Predictor Result**  
**Training Results:**  
- **Place to CTS Model:**  
  - R² Score: ${result.place_to_cts.r2_score.toFixed(4)}  
  - MAE: ${result.place_to_cts.mae.toFixed(4)}  
  - MSE: ${result.place_to_cts.mse.toFixed(4)}  `;

      if (result.combined_to_route) {
        trainingMessage += `
- **Combined to Route Model:**  
  - R² Score: ${result.combined_to_route.r2_score.toFixed(4)}  
  - MAE: ${result.combined_to_route.mae.toFixed(4)}  
  - MSE: ${result.combined_to_route.mse.toFixed(4)}  `;
      } else {
        trainingMessage += `
- **Combined to Route Model:** Not available (route table missing)  `;
      }

      const event = new CustomEvent('addPredictorMessage', {
        detail: {
          message: {
            id: `predictor-train-${Date.now()}`,
            role: 'assistant',
            content: trainingMessage,
            timestamp: new Date(),
            predictor: true,
            isServerResponse: true,
            isPredictorResult: true, // Added to identify predictor results
          },
        },
      });
      window.dispatchEvent(event);

      if (onTrainingComplete) {
        onTrainingComplete(result);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);

      const event = new CustomEvent('addPredictorMessage', {
        detail: {
          message: {
            id: `predictor-error-${Date.now()}`,
            role: 'assistant',
            content: `**Predictor Result**  
**Error:** ${errorMessage}`,
            timestamp: new Date(),
            predictor: true,
            isServerResponse: true,
            isPredictorResult: true, // Added to identify predictor errors
            error: errorMessage,
          },
        },
      });
      window.dispatchEvent(event);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      p={4}
      bg="gray.800"
      borderRadius="md"
      boxShadow="md"
      maxW="500px"
      mx="auto"
      mb={4}
    >
      <Heading as="h3" size="md" mb={4} color="white">
        Train Predictor Model
      </Heading>
      <form onSubmit={handleSubmit}>
        <FormControl mb={4}>
          <FormLabel color="gray.300">Place Table Name</FormLabel>
          <Input
            type="text"
            value={placeTable}
            onChange={(e) => setPlaceTable(e.target.value)}
            isDisabled={loading}
            bg="gray.700"
            color="white"
            borderColor="gray.600"
            _hover={{ borderColor: 'gray.500' }}
          />
        </FormControl>
        <FormControl mb={4}>
          <FormLabel color="gray.300">CTS Table Name</FormLabel>
          <Input
            type="text"
            value={ctsTable}
            onChange={(e) => setCtsTable(e.target.value)}
            isDisabled={loading}
            bg="gray.700"
            color="white"
            borderColor="gray.600"
            _hover={{ borderColor: 'gray.500' }}
          />
        </FormControl>
        <FormControl mb={4}>
          <FormLabel color="gray.300">Route Table Name (Optional)</FormLabel>
          <Input
            type="text"
            value={routeTable}
            onChange={(e) => setRouteTable(e.target.value)}
            placeholder="Leave empty if not available"
            isDisabled={loading}
            bg="gray.700"
            color="white"
            borderColor="gray.600"
            _hover={{ borderColor: 'gray.500' }}
          />
        </FormControl>
        {error && (
          <Text color="red.400" fontSize="sm" mb={4}>
            {error}
          </Text>
        )}
        <Button
          type="submit"
          isLoading={loading}
          isDisabled={loading}
          colorScheme="teal"
          width="full"
        >
          {loading ? 'Training...' : 'Train Model'}
        </Button>
      </form>
    </Box>
  );
};

export default TrainingForm;