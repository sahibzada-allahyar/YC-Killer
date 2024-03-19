import { Router, Request, Response } from 'express';
import { StatusCodes } from 'http-status-codes';
import * as Joi from 'joi';

import { commandProcessor } from '../services/command-processor';
import { validateRequest } from '../middleware';
import { ValidationError } from '../utils/errors';
import { routeLogger as logger } from '../utils/logger';

const router = Router();

// Validation schemas
const commandSchema = Joi.object({
  text: Joi.string().required().min(3).max(1000),
  context: Joi.object({
    userId: Joi.string().required(),
    accountId: Joi.string().optional(),
    transactionId: Joi.string().optional(),
    metadata: Joi.object().optional(),
  }).required(),
});

const audioCommandSchema = Joi.object({
  audio: Joi.binary().required(),
  context: Joi.object({
    userId: Joi.string().required(),
    accountId: Joi.string().optional(),
    transactionId: Joi.string().optional(),
    metadata: Joi.object().optional(),
  }).required(),
});

const suggestionsSchema = Joi.object({
  userId: Joi.string().required(),
  context: Joi.string().optional(),
});

/**
 * @swagger
 * /api/commands:
 *   post:
 *     summary: Process a natural language command
 *     tags: [Commands]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - text
 *               - context
 *             properties:
 *               text:
 *                 type: string
 *                 description: The natural language command
 *               context:
 *                 type: object
 *                 required:
 *                   - userId
 *                 properties:
 *                   userId:
 *                     type: string
 *                   accountId:
 *                     type: string
 *                   transactionId:
 *                     type: string
 *                   metadata:
 *                     type: object
 */
router.post('/', validateRequest(commandSchema), async (req: Request, res: Response) => {
  const { text, context } = req.body;
  const result = await commandProcessor.processCommand(text, context);

  res.status(StatusCodes.OK).json({
    success: true,
    data: result,
  });
});

/**
 * @swagger
 * /api/commands/audio:
 *   post:
 *     summary: Process a voice command
 *     tags: [Commands]
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             required:
 *               - audio
 *               - context
 *             properties:
 *               audio:
 *                 type: string
 *                 format: binary
 *               context:
 *                 type: object
 *                 required:
 *                   - userId
 *                 properties:
 *                   userId:
 *                     type: string
 *                   accountId:
 *                     type: string
 *                   transactionId:
 *                     type: string
 *                   metadata:
 *                     type: object
 */
router.post('/audio', validateRequest(audioCommandSchema), async (req: Request, res: Response) => {
  const { audio, context } = req.body;

  if (!Buffer.isBuffer(audio)) {
    throw new ValidationError('Audio data must be provided as a buffer');
  }

  // Transcribe audio to text
  const text = await commandProcessor.transcribeAudio(audio);

  // Process the transcribed command
  const result = await commandProcessor.processCommand(text, context);

  res.status(StatusCodes.OK).json({
    success: true,
    data: {
      transcription: text,
      result,
    },
  });
});

/**
 * @swagger
 * /api/commands/suggestions:
 *   get:
 *     summary: Get command suggestions for a user
 *     tags: [Commands]
 *     parameters:
 *       - in: query
 *         name: userId
 *         required: true
 *         schema:
 *           type: string
 *       - in: query
 *         name: context
 *         schema:
 *           type: string
 */
router.get('/suggestions', validateRequest(suggestionsSchema, 'query'), async (req: Request, res: Response) => {
  const { userId, context } = req.query as { userId: string; context?: string };
  const suggestions = await commandProcessor.generateSuggestions(userId, context);

  res.status(StatusCodes.OK).json({
    success: true,
    data: suggestions,
  });
});

export const commandRoutes = router;
