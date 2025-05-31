import React, { useState, useEffect } from 'react';
import { useToast } from '@chakra-ui/react';
import { getUserAIRules, saveAIRule, AIRule } from '../../services/aiRulesService';

const AIRulesSettings: React.FC = () => {
  const [ruleContent, setRuleContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [currentRule, setCurrentRule] = useState<AIRule | null>(null);
  const toast = useToast();

  // Load existing rule on component mount
  useEffect(() => {
    const loadRule = async () => {
      try {
        setIsLoading(true);
        const rules = await getUserAIRules();
        if (rules.length > 0) {
          setCurrentRule(rules[0]);
          setRuleContent(rules[0].rule_content);
        }
      } catch (error) {
        toast({
          title: 'Error loading AI rules',
          description: 'Could not load your AI rules. Please try again later.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadRule();
  }, [toast]);

  // Handle saving the rule
  const handleSave = async () => {
    try {
      setIsSaving(true);
      const savedRule = await saveAIRule(ruleContent);
      setCurrentRule(savedRule);

      toast({
        title: 'AI rule saved',
        description: 'Your AI rule has been saved successfully.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Error saving AI rule',
        description: 'Could not save your AI rule. Please try again later.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="w-full p-5 border border-solid rounded-lg" style={{
      borderColor: 'var(--color-border)',
      backgroundColor: 'var(--color-surface)',
      color: 'var(--color-text)'
    }}>
      <div className="flex flex-col space-y-4">
        <h3 className="text-lg font-semibold" style={{ color: 'var(--color-text)' }}>AI Rules</h3>

        <p style={{ color: 'var(--color-text-secondary)' }}>
          These preferences get sent to the AI on all chats, composers and Ctrl-K sessions.
          The AI will read this context when needed to better understand your preferences.
        </p>

        {isLoading ? (
          <div className="text-center py-4">
            <div className="inline-block animate-spin rounded-full h-6 w-6 border-t-2 border-b-2" style={{ borderColor: 'var(--color-primary)' }}></div>
            <p className="mt-2" style={{ color: 'var(--color-text-secondary)' }}>Loading your AI rules...</p>
          </div>
        ) : (
          <div className="mb-4">
            <label className="block mb-2 font-medium" style={{ color: 'var(--color-text)' }}>User Rules</label>
            <textarea
              value={ruleContent}
              onChange={(e) => setRuleContent(e.target.value)}
              placeholder="e.g., 'Always output answers in Portuguese, prefer terse answers, always use functional code'"
              className="w-full p-3 rounded-md border"
              style={{
                backgroundColor: 'var(--color-input-bg)',
                color: 'var(--color-text)',
                borderColor: 'var(--color-border)'
              }}
              rows={6}
            />
          </div>
        )}

        <div className="flex justify-end">
          <button
            onClick={handleSave}
            disabled={isLoading}
            className="px-4 py-2 rounded-md text-white font-medium"
            style={{
              backgroundColor: isSaving ? 'var(--color-primary-muted)' : 'var(--color-primary)',
              cursor: isLoading ? 'not-allowed' : 'pointer'
            }}
          >
            {isSaving ? 'Saving...' : 'Save'}
          </button>
        </div>

        {currentRule && (
          <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
            Last updated: {new Date(currentRule.updated_at).toLocaleString()}
          </p>
        )}
      </div>
    </div>
  );
};

export default AIRulesSettings;
