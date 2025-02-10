import { InvestigationProgress } from './types/research.js';

export class ProgressManager {
  private currentProgress: InvestigationProgress | null = null;
  private lastUpdateTime = 0;
  private readonly updateInterval = 500; // Minimum time between updates in ms

  updateProgress(progress: InvestigationProgress): void {
    const now = Date.now();
    if (now - this.lastUpdateTime < this.updateInterval) {
      return; // Throttle updates
    }

    this.currentProgress = progress;
    this.lastUpdateTime = now;
    this.renderProgress();
  }

  private renderProgress(): void {
    if (!this.currentProgress) return;

    const {
      currentLevel,
      totalLevels,
      currentScope,
      totalScope,
      currentInquiry,
      completedInquiries,
      totalInquiries,
    } = this.currentProgress;

    // Clear previous line
    process.stdout.write('\r\x1b[K');

    // Build progress string
    const levelProgress = `Level ${currentLevel}/${totalLevels}`;
    const scopeProgress = `Scope ${currentScope}/${totalScope}`;
    const inquiryProgress = `Queries ${completedInquiries}/${totalInquiries}`;
    
    let progressString = `${levelProgress} | ${scopeProgress} | ${inquiryProgress}`;
    
    if (currentInquiry) {
      // Truncate inquiry if too long
      const maxInquiryLength = process.stdout.columns 
        ? Math.max(process.stdout.columns - progressString.length - 5, 20)
        : 50;
      const truncatedInquiry = currentInquiry.length > maxInquiryLength
        ? currentInquiry.substring(0, maxInquiryLength - 3) + '...'
        : currentInquiry;
      progressString += ` | Current: ${truncatedInquiry}`;
    }

    process.stdout.write(progressString);
  }
}
