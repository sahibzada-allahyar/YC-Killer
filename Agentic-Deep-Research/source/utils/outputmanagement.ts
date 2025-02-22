export class OutputManager {
  private static instance: OutputManager;
  private messageBuffer: string[] = [];
  private readonly bufferLimit = 1000;

  constructor() {
    if (OutputManager.instance) {
      return OutputManager.instance;
    }
    OutputManager.instance = this;
  }

  log(...args: any[]) {
    const message = args
      .map(arg =>
        typeof arg === 'string' ? arg : JSON.stringify(arg, null, 2),
      )
      .join(' ');

    // Add to buffer with timestamp
    const timestamp = new Date().toISOString();
    this.messageBuffer.push(`[${timestamp}] ${message}`);

    // Trim buffer if it exceeds limit
    if (this.messageBuffer.length > this.bufferLimit) {
      this.messageBuffer = this.messageBuffer.slice(-this.bufferLimit);
    }

    // Print to console
    console.log(...args);
  }

  getBuffer(): string[] {
    return [...this.messageBuffer];
  }

  clearBuffer() {
    this.messageBuffer = [];
  }
}
