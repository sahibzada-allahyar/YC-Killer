import { Router, Request, Response } from 'express';
import { StatusCodes } from 'http-status-codes';
import * as Joi from 'joi';

import { accountingService } from '../services/accounting/accounting-service';
import { validateRequest } from '../middleware';
import { NotFoundError, ValidationError } from '../utils/errors';
import { routeLogger as logger } from '../utils/logger';

const router = Router();

// Validation schemas
const transactionSchema = Joi.object({
  date: Joi.date().required().max('now').messages({
    'date.max': 'Transaction date cannot be in the future',
  }),
  description: Joi.string().required().min(3).max(500),
  amount: Joi.number().required().messages({
    'number.base': 'Amount must be a valid number',
  }),
  type: Joi.string()
    .required()
    .valid('debit', 'credit')
    .messages({
      'any.only': 'Transaction type must be either debit or credit',
    }),
  accountId: Joi.string().required(),
  category: Joi.string().required(),
  reference: Joi.string().optional(),
  metadata: Joi.object().optional(),
});

const transactionQuerySchema = Joi.object({
  startDate: Joi.date().optional(),
  endDate: Joi.date().optional().min(Joi.ref('startDate')).messages({
    'date.min': 'End date must be after start date',
  }),
  accountId: Joi.string().optional(),
  category: Joi.string().optional(),
  type: Joi.string().optional().valid('debit', 'credit'),
  minAmount: Joi.number().optional(),
  maxAmount: Joi.number().optional().min(Joi.ref('minAmount')).messages({
    'number.min': 'Maximum amount must be greater than minimum amount',
  }),
  limit: Joi.number().optional().min(1).max(100).default(50),
  offset: Joi.number().optional().min(0).default(0),
});

/**
 * @swagger
 * /api/transactions:
 *   get:
 *     summary: Get transactions with optional filtering
 *     tags: [Transactions]
 *     parameters:
 *       - in: query
 *         name: startDate
 *         schema:
 *           type: string
 *           format: date
 *       - in: query
 *         name: endDate
 *         schema:
 *           type: string
 *           format: date
 *       - in: query
 *         name: accountId
 *         schema:
 *           type: string
 *       - in: query
 *         name: category
 *         schema:
 *           type: string
 *       - in: query
 *         name: type
 *         schema:
 *           type: string
 *           enum: [debit, credit]
 *       - in: query
 *         name: minAmount
 *         schema:
 *           type: number
 *       - in: query
 *         name: maxAmount
 *         schema:
 *           type: number
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 100
 *           default: 50
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           minimum: 0
 *           default: 0
 */
router.get('/', validateRequest(transactionQuerySchema, 'query'), async (req: Request, res: Response) => {
  const transactions = await accountingService.getTransactions(req.query);
  res.status(StatusCodes.OK).json({
    success: true,
    data: transactions,
  });
});

/**
 * @swagger
 * /api/transactions/{id}:
 *   get:
 *     summary: Get transaction by ID
 *     tags: [Transactions]
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 */
router.get('/:id', async (req: Request, res: Response) => {
  const transactionId = req.params['id'];
  if (!transactionId) {
    throw new ValidationError('Transaction ID is required');
  }

  const transaction = await accountingService.getTransaction(transactionId);
  if (!transaction) {
    throw new NotFoundError(`Transaction with ID ${transactionId} not found`);
  }

  res.status(StatusCodes.OK).json({
    success: true,
    data: transaction,
  });
});

/**
 * @swagger
 * /api/transactions:
 *   post:
 *     summary: Create a new transaction
 *     tags: [Transactions]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/Transaction'
 */
router.post('/', validateRequest(transactionSchema), async (req: Request, res: Response) => {
  // Verify account exists
  const account = await accountingService.getAccount(req.body.accountId);
  if (!account) {
    throw new NotFoundError(`Account with ID ${req.body.accountId} not found`);
  }

  const transaction = await accountingService.createTransaction(req.body);
  res.status(StatusCodes.CREATED).json({
    success: true,
    data: transaction,
  });
});

/**
 * @swagger
 * /api/transactions/{id}:
 *   put:
 *     summary: Update a transaction
 *     tags: [Transactions]
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 */
router.put('/:id', validateRequest(transactionSchema), async (req: Request, res: Response) => {
  const transactionId = req.params['id'];
  if (!transactionId) {
    throw new ValidationError('Transaction ID is required');
  }

  // Verify account exists
  const account = await accountingService.getAccount(req.body.accountId);
  if (!account) {
    throw new NotFoundError(`Account with ID ${req.body.accountId} not found`);
  }

  const transaction = await accountingService.updateTransaction(transactionId, req.body);
  if (!transaction) {
    throw new NotFoundError(`Transaction with ID ${transactionId} not found`);
  }

  res.status(StatusCodes.OK).json({
    success: true,
    data: transaction,
  });
});

/**
 * @swagger
 * /api/transactions/{id}:
 *   delete:
 *     summary: Delete a transaction
 *     tags: [Transactions]
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 */
router.delete('/:id', async (req: Request, res: Response) => {
  const transactionId = req.params['id'];
  if (!transactionId) {
    throw new ValidationError('Transaction ID is required');
  }

  await accountingService.deleteTransaction(transactionId);
  res.status(StatusCodes.NO_CONTENT).send();
});

/**
 * @swagger
 * /api/transactions/categorize:
 *   post:
 *     summary: Categorize transactions using AI
 *     tags: [Transactions]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               transactionIds:
 *                 type: array
 *                 items:
 *                   type: string
 */
router.post(
  '/categorize',
  validateRequest(
    Joi.object({
      transactionIds: Joi.array().items(Joi.string()).required().min(1),
    })
  ),
  async (req: Request, res: Response) => {
    const { transactionIds } = req.body;
    const results = await accountingService.categorizeTransactions(transactionIds);

    res.status(StatusCodes.OK).json({
      success: true,
      data: results,
    });
  }
);

export const transactionRoutes = router;
