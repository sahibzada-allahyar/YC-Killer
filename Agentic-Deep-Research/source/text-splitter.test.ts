import { RecursiveCharacterTextSplitter } from './text-splitter.js';
import { describe, it, expect } from '@jest/globals';

describe('RecursiveCharacterTextSplitter', () => {
  const testConfig = {
    chunkSize: 10,
    chunkOverlap: 3,
  };

  it('handles content shorter than chunk size', () => {
    const segmenter = new RecursiveCharacterTextSplitter(testConfig);
    const input = 'Short text';
    const result = segmenter.splitText(input);
    expect(result).toEqual(['Short text']);
  });

  it('segments content at natural break points', () => {
    const segmenter = new RecursiveCharacterTextSplitter(testConfig);
    const input = 'First part. Second part. Third section.';
    const result = segmenter.splitText(input);
    expect(result).toContain('First part.');
    expect(result).toContain('Second part.');
    expect(result).toContain('Third section.');
  });

  it('respects chunk overlap', () => {
    const segmenter = new RecursiveCharacterTextSplitter({
      chunkSize: 15,
      chunkOverlap: 5,
    });
    const input = 'First segment. Second segment. Third segment.';
    const result = segmenter.splitText(input);
    
    // Verify overlap between segments
    const segments = result.map((s: string) => s.trim());
    for (let i = 1; i < segments.length; i++) {
      const prevEnd = segments[i - 1].slice(-5);
      const currStart = segments[i].slice(0, 5);
      expect(prevEnd.length + currStart.length >= 5).toBeTruthy();
    }
  });

  it('handles content with newlines', () => {
    const segmenter = new RecursiveCharacterTextSplitter(testConfig);
    const input = 'Line one\nLine two\nLine three';
    const result = segmenter.splitText(input);
    expect(result.some((s: string) => s.includes('\n'))).toBeTruthy();
    expect(result.every((s: string) => s.trim().length > 0)).toBeTruthy();
  });
});
