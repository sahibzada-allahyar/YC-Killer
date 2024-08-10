import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { HealthAgentOrchestrator } from './orchestrator/healthAgentOrchestrator';
import { PatientQuery } from './types/agents';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Initialize the health agent orchestrator
const orchestrator = new HealthAgentOrchestrator();

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', agents: orchestrator.getAgents().map(a => a.name) });
});

// Main query endpoint
app.post('/query', async (req, res) => {
  try {
    const query: PatientQuery = req.body;
    
    if (!query.query) {
      return res.status(400).json({ 
        error: 'Query is required' 
      });
    }

    const response = await orchestrator.processQuery(query);
    res.json(response);
  } catch (error) {
    console.error('Error processing query:', error);
    res.status(500).json({ 
      error: 'Failed to process query',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Error handling middleware
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ 
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'An unexpected error occurred'
  });
});

// Start server
app.listen(port, () => {
  console.log(`Healthcare Agent Server running on port ${port}`);
  console.log('Available agents:', orchestrator.getAgents().map(a => a.name).join(', '));
});

export default app;
