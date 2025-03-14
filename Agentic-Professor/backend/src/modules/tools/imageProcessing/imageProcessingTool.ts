import { createWorker } from 'tesseract.js';
import { ITool } from '../tool.interface';

interface ImageProcessingConfig {
  defaultLanguage: string;
}

interface ImageProcessingParams {
  imageData: string;
  language?: string;
  enhanceImage?: boolean;
}

interface ProcessedImage {
  text: string;
  confidence: number;
  language: string;
}

export class ImageProcessingTool implements ITool {
  public name = 'imageProcessing';
  public description = 'Process and extract text from images using OCR';
  public parameters = {
    properties: {
      imageData: {
        type: 'string',
        description: 'Base64 encoded image data'
      },
      language: {
        type: 'string',
        description: 'Language code for OCR (e.g., eng, fra, deu)',
        default: 'eng'
      },
      enhanceImage: {
        type: 'boolean',
        description: 'Whether to apply image enhancement before OCR',
        default: false
      }
    },
    required: ['imageData']
  };

  private worker: Awaited<ReturnType<typeof createWorker>> | null = null;
  private currentLanguage: string;

  constructor(private config: ImageProcessingConfig) {
    this.currentLanguage = config.defaultLanguage;
  }

  private async initWorker(language: string): Promise<void> {
    if (!this.worker || this.currentLanguage !== language) {
      if (this.worker) {
        await this.worker.terminate();
      }

      this.worker = await createWorker();
      await this.worker.loadLanguage(language);
      await this.worker.initialize(language);
      this.currentLanguage = language;
    }
  }

  private async enhanceImage(imageData: string): Promise<string> {
    // In a real implementation, you might:
    // 1. Convert base64 to image
    // 2. Apply preprocessing (e.g., contrast enhancement, noise reduction)
    // 3. Convert back to base64
    // For now, we'll just return the original image
    return imageData;
  }

  public async execute(input: string): Promise<ProcessedImage> {
    try {
      const params: ImageProcessingParams = JSON.parse(input);
      const language = params.language || this.config.defaultLanguage;

      let processedImageData = params.imageData;
      if (params.enhanceImage) {
        processedImageData = await this.enhanceImage(params.imageData);
      }

      await this.initWorker(language);
      if (!this.worker) {
        throw new Error('Failed to initialize OCR worker');
      }

      const result = await this.worker.recognize(processedImageData);

      return {
        text: result.data.text,
        confidence: result.data.confidence,
        language
      };
    } catch (error) {
      throw new Error(
        `Failed to process image: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  public async cleanup(): Promise<void> {
    if (this.worker) {
      await this.worker.terminate();
      this.worker = null;
    }
  }
}
