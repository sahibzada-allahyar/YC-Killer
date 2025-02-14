import { InvestigationProgress } from './types/research';

export class ProgressManager {
  private lastUpdate: number = 0;
  private readonly updateInterval: number = 100; // ms

  updateProgress(progress: InvestigationProgress) {
    const now = Date.now();
    if (now - this.lastUpdate < this.updateInterval) {
      return;
    }
    this.lastUpdate = now;

    const depthProgress = `Depth: ${progress.currentLevel}/${progress.totalLevels}`;
    const breadthProgress = `Width: ${progress.currentScope}/${progress.totalScope}`;
    const queryProgress = `Queries: ${progress.completedInquiries}/${progress.totalInquiries}`;
    
    const currentQuery = progress.currentInquiry 
      ? `\nCurrent Query: ${progress.currentInquiry}`
      : '';

    // Use stdout.write to update progress in place
    process.stdout.clearLine(0);
    process.stdout.cursorTo(0);
    process.stdout.write(
      `${depthProgress} | ${breadthProgress} | ${queryProgress}${currentQuery}`,
    );
  }
}
