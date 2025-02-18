export type InvestigationProgress = {
  currentLevel: number;
  totalLevels: number;
  currentScope: number;
  totalScope: number;
  currentInquiry?: string;
  totalInquiries: number;
  completedInquiries: number;
};

export type InvestigationOutcome = {
  insights: string[];
  exploredSources: string[];
};

export type SearchQueryMetadata = {
  inquiry: string;
  investigationIntent: string;
};

export type ProcessedInsight = {
  insights: string[];
  followUpInquiries: string[];
};

export type ReportInput = {
  initialQuery: string;
  insights: string[];
  exploredSources: string[];
};
