import express from 'express';
import { callController } from '../controllers/callController';
import { authenticateAdmin } from '../middleware/auth';

const router = express.Router();

// Twilio webhook endpoints
router.post('/incoming', callController.handleIncoming);
router.post('/outbound', callController.handleOutbound);
router.post('/response', callController.handleResponse);
router.post('/complete', callController.handleComplete);
router.post('/recording-status', callController.handleRecordingStatus);

// REST API endpoints (require authentication)
router.post('/initiate', authenticateAdmin, callController.initiateOutboundCall);
router.get('/recording/:callSid', authenticateAdmin, callController.getRecording);

export default router;
