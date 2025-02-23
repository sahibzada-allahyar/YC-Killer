import { createOpenAI, type OpenAIProviderSettings } from '@ai-sdk/openai';
import { getEncoding } from 'js-tiktoken';
import { RecursiveCharacterTextSplitter } from './text-splitter.js';

interface EnhancedOpenAISettings extends OpenAIProviderSettings {
  baseURL?: string;
}

// Initialize AI Provider
const aiProvider = createOpenAI({
  apiKey: process.env.OPENAI_KEY!,
  baseURL: process.env.OPENAI_ENDPOINT || 'https://api.openai.com/v1',
} as EnhancedOpenAISettings);

const selectedModel = process.env.OPENAI_MODEL || 'o3-mini';

// Configure Model Settings
export const knowledgeProcessor = aiProvider(selectedModel, {
  reasoningEffort: selectedModel.startsWith('o') ? 'medium' : undefined,
  structuredOutputs: true,
});

const MinSegmentSize = 140;
const tokenEncoder = getEncoding('o200k_base');

export function optimizePromptLength(
  content: string,
  maxTokens = Number(process.env.CONTEXT_SIZE) || 128_000,
): string {
  if (!content) {
    return '';
  }

  const tokenCount = tokenEncoder.encode(content).length;
  if (tokenCount <= maxTokens) {
    return content;
  }

  const excessTokens = tokenCount - maxTokens;
  // Approximate character count based on token ratio (typically 3 chars per token)
  const targetLength = content.length - excessTokens * 3;
  
  if (targetLength < MinSegmentSize) {
    return content.slice(0, MinSegmentSize);
  }

  const segmenter = new RecursiveCharacterTextSplitter({
    chunkSize: targetLength,
    chunkOverlap: 0,
  });
  
  const optimizedContent = segmenter.splitText(content)[0] ?? '';

  // Handle edge case where segmentation didn't reduce length
  if (optimizedContent.length === content.length) {
    return optimizePromptLength(content.slice(0, targetLength), maxTokens);
  }

  // Recursively optimize until within token limit
  return optimizePromptLength(optimizedContent, maxTokens);
}
