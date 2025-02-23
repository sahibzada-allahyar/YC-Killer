export interface SegmentationConfig {
  chunkSize: number;
  chunkOverlap: number;
}

export class RecursiveCharacterTextSplitter {
  private readonly chunkSize: number;
  private readonly chunkOverlap: number;

  constructor(config: SegmentationConfig) {
    this.chunkSize = config.chunkSize;
    this.chunkOverlap = config.chunkOverlap;
  }

  splitText(content: string): string[] {
    if (content.length <= this.chunkSize) {
      return [content];
    }

    const segments: string[] = [];
    let startIndex = 0;

    while (startIndex < content.length) {
      let endIndex = startIndex + this.chunkSize;
      
      if (endIndex > content.length) {
        endIndex = content.length;
      } else {
        // Try to find a natural break point
        const lastPeriod = content.lastIndexOf('.', endIndex);
        const lastNewline = content.lastIndexOf('\n', endIndex);
        const naturalBreak = Math.max(lastPeriod, lastNewline);
        
        if (naturalBreak > startIndex) {
          endIndex = naturalBreak + 1;
        }
      }

      segments.push(content.slice(startIndex, endIndex).trim());
      startIndex = endIndex - this.chunkOverlap;
    }

    return segments;
  }
}
