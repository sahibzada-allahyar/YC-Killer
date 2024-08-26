"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const dotenv_1 = __importDefault(require("dotenv"));
const healthAgentOrchestrator_1 = require("./orchestrator/healthAgentOrchestrator");
dotenv_1.default.config();
const app = (0, express_1.default)();
const port = process.env.PORT || 3000;
// Initialize the health agent orchestrator
const orchestrator = new healthAgentOrchestrator_1.HealthAgentOrchestrator();
// Middleware
app.use((0, cors_1.default)());
app.use(express_1.default.json());
// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', agents: orchestrator.getAgents().map(a => a.name) });
});
// Main query endpoint
app.post('/query', async (req, res) => {
    try {
        const query = req.body;
        if (!query.query) {
            return res.status(400).json({
                error: 'Query is required'
            });
        }
        const response = await orchestrator.processQuery(query);
        res.json(response);
    }
    catch (error) {
        console.error('Error processing query:', error);
        res.status(500).json({
            error: 'Failed to process query',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
// Error handling middleware
app.use((err, req, res, next) => {
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
exports.default = app;
