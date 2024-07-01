interface MockProcedure {
  keywords: string[];
  covered: boolean;
  estimatedCoverage: number;
  notes: string;
  historicalApprovalRate?: number;
  specialistReferralRequired?: boolean;
  annualLimit?: number;
  category: 'preventive' | 'diagnostic' | 'treatment' | 'specialist' | 'therapy';
}

// Enhanced mock dataset with broader medical coverage
const mockProcedureDatabase: MockProcedure[] = [
  // Preventive Care
  {
    keywords: ['screening', 'preventive', 'checkup', 'annual exam'],
    covered: true,
    estimatedCoverage: 100,
    notes: 'Preventive care fully covered under ACA guidelines',
    historicalApprovalRate: 98,
    category: 'preventive'
  },
  // Diagnostic Tests
  {
    keywords: ['blood test', 'lab work', 'imaging', 'x-ray', 'mri', 'scan'],
    covered: true,
    estimatedCoverage: 80,
    notes: 'Most diagnostic tests covered with pre-authorization',
    historicalApprovalRate: 85,
    category: 'diagnostic'
  },
  // Chronic Conditions
  {
    keywords: ['diabetes', 'blood sugar', 'glucose', 'a1c'],
    covered: true,
    estimatedCoverage: 80,
    notes: 'Chronic condition management covered under preventive care',
    historicalApprovalRate: 90,
    category: 'treatment'
  },
  // Specialist Consultations
  {
    keywords: ['specialist', 'consultation', 'referral'],
    covered: true,
    estimatedCoverage: 70,
    notes: 'Specialist visits covered with referral',
    specialistReferralRequired: true,
    historicalApprovalRate: 75,
    category: 'specialist'
  },
  // Medications
  {
    keywords: ['medication', 'prescription', 'drug', 'medicine'],
    covered: true,
    estimatedCoverage: 60,
    notes: 'Prescription coverage through pharmacy benefit',
    historicalApprovalRate: 82,
    category: 'treatment'
  },
  // Therapy Services
  {
    keywords: ['therapy', 'counseling', 'rehabilitation', 'physical therapy'],
    covered: true,
    estimatedCoverage: 65,
    notes: 'Therapy services covered with annual visit limits',
    annualLimit: 20,
    historicalApprovalRate: 88,
    category: 'therapy'
  },
  // Emergency Care
  {
    keywords: ['emergency', 'urgent', 'immediate care'],
    covered: true,
    estimatedCoverage: 85,
    notes: 'Emergency services covered at higher rate',
    historicalApprovalRate: 95,
    category: 'treatment'
  }
];

// Mock historical approval data
const mockHistoricalData = {
  getApprovalRate(category: MockProcedure['category']): number {
    const procedures = mockProcedureDatabase.filter(p => p.category === category);
    if (procedures.length === 0) return 0;
    return procedures.reduce((sum, p) => sum + (p.historicalApprovalRate || 0), 0) / procedures.length;
  },
  
  hasAnnualLimitRemaining(category: MockProcedure['category']): boolean {
    const procedure = mockProcedureDatabase.find(p => p.category === category);
    return procedure?.annualLimit ? Math.random() < 0.8 : true; // 80% chance of having visits remaining
  }
};

export interface InsuranceCoverageInfo {
  covered: boolean;
  estimatedCoverage: number;
  notes: string;
  historicalApprovalRate?: number;
  requiresPreAuth?: boolean;
  annualLimitRemaining?: boolean;
}

export class MockInsuranceService {
  checkCoverage(text: string): InsuranceCoverageInfo {
    const lowerText = text.toLowerCase();
    
    // Find matching procedures
    const matchingProcedures = mockProcedureDatabase.filter(procedure =>
      procedure.keywords.some(keyword => lowerText.includes(keyword.toLowerCase()))
    );

    if (matchingProcedures.length === 0) {
      return {
        covered: false,
        estimatedCoverage: 0,
        notes: 'No coverage information found for the specified treatment or condition',
        historicalApprovalRate: 0
      };
    }

    // If multiple matches, prioritize by coverage and historical approval rate
    const bestCoverage = matchingProcedures.reduce((best, current) => {
      const currentScore = (current.estimatedCoverage + (current.historicalApprovalRate || 0)) / 2;
      const bestScore = (best.estimatedCoverage + (best.historicalApprovalRate || 0)) / 2;
      return currentScore > bestScore ? current : best;
    });

    const annualLimitRemaining = mockHistoricalData.hasAnnualLimitRemaining(bestCoverage.category);
    const historicalApprovalRate = mockHistoricalData.getApprovalRate(bestCoverage.category);

    return {
      covered: bestCoverage.covered,
      estimatedCoverage: bestCoverage.estimatedCoverage,
      notes: bestCoverage.notes + (bestCoverage.specialistReferralRequired ? ' (Requires referral)' : ''),
      historicalApprovalRate,
      requiresPreAuth: bestCoverage.category === 'diagnostic' || bestCoverage.specialistReferralRequired,
      annualLimitRemaining
    };
  }
}
