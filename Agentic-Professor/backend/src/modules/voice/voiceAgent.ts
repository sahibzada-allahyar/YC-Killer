import OpenAI from 'openai';
import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import { Config } from '@ai-tutor/shared';

interface VoiceResponse {
  audioUrl: string;
}

export class VoiceAgent {
  private openai: OpenAI;
  private cacheDir: string;

  constructor(apiKey: string, private config: Config) {
    this.openai = new OpenAI({ apiKey });
    this.cacheDir = path.join(process.cwd(), 'cache', 'voice');
  }

  private generateCacheKey(text: string): string {
    return crypto.createHash('md5').update(text).digest('hex');
  }

  private async getCachedAudio(cacheKey: string): Promise<string | null> {
    try {
      const filePath = path.join(this.cacheDir, `${cacheKey}.mp3`);
      await fs.access(filePath);
      return `/cache/voice/${cacheKey}.mp3`;
    } catch {
      return null;
    }
  }

  private async cacheAudio(cacheKey: string, audioData: Buffer): Promise<string> {
    const filePath = path.join(this.cacheDir, `${cacheKey}.mp3`);
    await fs.writeFile(filePath, audioData);
    return `/cache/voice/${cacheKey}.mp3`;
  }

  public async speak(text: string): Promise<VoiceResponse> {
    try {
      const cacheKey = this.generateCacheKey(text);
      const cachedUrl = await this.getCachedAudio(cacheKey);

      if (cachedUrl) {
        return { audioUrl: cachedUrl };
      }

      const response = await this.openai.audio.speech.create({
        model: 'tts-1',
        voice: 'alloy',
        input: text
      });

      const audioData = Buffer.from(await response.arrayBuffer());
      const audioUrl = await this.cacheAudio(cacheKey, audioData);

      return { audioUrl };
    } catch (error) {
      throw new Error(
        `Voice synthesis error: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  public async cleanup(): Promise<void> {
    // Implement any cleanup needed
  }
}
