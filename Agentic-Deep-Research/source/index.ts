import { conductInvestigation, generateAnalysisReport } from './core/comprehensive-investigation.js';
import type { InvestigationProgress, InvestigationOutcome } from './types/research.js';

export {
  conductInvestigation,
  generateAnalysisReport,
  type InvestigationProgress,
  type InvestigationOutcome,
};

// Re-export types
export type { SearchData, SearchItem } from './types/search.js';
