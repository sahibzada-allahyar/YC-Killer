import { InsuranceCoverageInfo } from '../services/mockInsurance';

export type ComplexityLevel = 'low' | 'moderate' | 'high';

export interface HealthResponse {
  message: string;
  recommendations: string[];
  priority: 'low' | 'medium' | 'high';
  insuranceCoverage?: InsuranceCoverageInfo;
  complexity?: ComplexityLevel;
  reasoning?: string;
}

export interface CollaborationMessage {
  from: string;
  content: string;
  timestamp: string;
}

export interface TeamDiscussion {
  messages: CollaborationMessage[];
  conclusion?: string;
}

export interface HealthAgent {
  name: string;
  role: string;
  processQuery: (query: string) => Promise<HealthResponse>;
}

export interface PatientQuery {
  query: string;
  patientContext?: {
    age?: number;
    diabetesType?: '1' | '2';
    lastBloodSugar?: number;
    medications?: string[];
  };
}

export interface AgentResponse {
  agentName: string;
  response: HealthResponse;
}

export interface MultiAgentResponse {
  query: string;
  responses: AgentResponse[];
  timestamp: string;
  complexity: ComplexityLevel;
  teamDiscussion?: TeamDiscussion;
}
