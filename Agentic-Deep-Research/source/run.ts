import { writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { conductInvestigation, generateAnalysisReport } from './index.js';
import { ProgressManager } from './progress-manager.js';

const progressManager = new ProgressManager();

async function executeResearch(query: string) {
  try {
    const { insights, exploredSources } = await conductInvestigation({
      query,
      expansionWidth: 3,
      explorationDepth: 3,
      onProgress: progress => {
        progressManager.updateProgress(progress);
      },
    });

    const report = await generateAnalysisReport({
      initialQuery: query,
      insights,
      exploredSources,
    });

    // Use reports directory in Docker, fallback to current directory
    const outputDir = process.env.NODE_ENV === 'production' ? '/app/reports' : '.';
    await mkdir(outputDir, { recursive: true });
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputPath = join(outputDir, `report-${timestamp}.md`);
    
    await writeFile(outputPath, report, 'utf-8');
    console.log(`\nResearch completed. Report saved to ${outputPath}`);
  } catch (error) {
    console.error('\nResearch failed:', error);
    process.exit(1);
  }
}

// Get the research query from command line arguments
const query = process.argv.slice(2).join(' ');
if (!query) {
  console.error('Please provide a research query');
  process.exit(1);
}

executeResearch(query).catch(error => {
  console.error('\nUnexpected error:', error);
  process.exit(1);
});
