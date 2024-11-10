"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MockInsuranceService = void 0;
// Simple mock dataset of medical procedures and their coverage
const mockProcedureDatabase = [
    {
        keywords: ['blood sugar', 'glucose', 'a1c', 'diabetes test'],
        covered: true,
        estimatedCoverage: 80,
        notes: 'Standard diabetes monitoring covered under preventive care'
    },
    {
        keywords: ['foot exam', 'neuropathy', 'numbness'],
        covered: true,
        estimatedCoverage: 70,
        notes: 'Diabetic foot exams covered with standard copay'
    },
    {
        keywords: ['insulin', 'medication', 'prescription'],
        covered: true,
        estimatedCoverage: 60,
        notes: 'Prescription coverage applies after deductible'
    },
    {
        keywords: ['diet', 'nutrition', 'counseling'],
        covered: true,
        estimatedCoverage: 50,
        notes: 'Limited number of nutrition counseling sessions covered annually'
    },
    {
        keywords: ['exercise', 'physical therapy', 'rehabilitation'],
        covered: true,
        estimatedCoverage: 65,
        notes: 'Physical therapy covered with referral'
    }
];
class MockInsuranceService {
    checkCoverage(text) {
        // Convert text to lowercase for case-insensitive matching
        const lowerText = text.toLowerCase();
        // Find matching procedures
        const matchingProcedures = mockProcedureDatabase.filter(procedure => procedure.keywords.some(keyword => lowerText.includes(keyword.toLowerCase())));
        if (matchingProcedures.length === 0) {
            return {
                covered: false,
                estimatedCoverage: 0,
                notes: 'No coverage information found for the specified treatment or condition'
            };
        }
        // If multiple matches, return the one with highest coverage
        const bestCoverage = matchingProcedures.reduce((best, current) => current.estimatedCoverage > best.estimatedCoverage ? current : best);
        return {
            covered: bestCoverage.covered,
            estimatedCoverage: bestCoverage.estimatedCoverage,
            notes: bestCoverage.notes
        };
    }
}
exports.MockInsuranceService = MockInsuranceService;
