import { Router, Request, Response } from 'express';
import { StatusCodes } from 'http-status-codes';
import * as Joi from 'joi';

import { accountingService } from '../services/accounting/accounting-service';
import { validateRequest } from '../middleware';
import { NotFoundError, ValidationError } from '../utils/errors';
import { routeLogger as logger } from '../utils/logger';

const router = Router();

// Validation schemas
const accountSchema = Joi.object({
  name: Joi.string().required().min(3).max(100),
  type: Joi.string()
    .required()
    .valid('asset', 'liability', 'equity', 'revenue', 'expense')
    .messages({
      'any.only': 'Account type must be one of: asset, liability, equity, revenue, expense',
    }),
  description: Joi.string().optional().max(500),
  currency: Joi.string()
    .required()
    .length(3)
    .pattern(/^[A-Z]{3}$/)
    .messages({
      'string.pattern.base': 'Currency must be a valid ISO 4217 code (e.g., USD, EUR, GBP)',
    }), // ISO 4217 currency code
  parentAccountId: Joi.string().optional(),
});

/**
 * @swagger
 * /api/accounts:
 *   get:
 *     summary: Get all accounts
 *     tags: [Accounts]
 *     responses:
 *       200:
 *         description: List of accounts retrieved successfully
 */
router.get('/', async (req: Request, res: Response) => {
  const accounts = await accountingService.getAccounts();
  res.status(StatusCodes.OK).json({
    success: true,
    data: accounts,
  });
});

/**
 * @swagger
 * /api/accounts/{id}:
 *   get:
 *     summary: Get account by ID
 *     tags: [Accounts]
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 */
router.get('/:id', async (req: Request, res: Response) => {
  const accountId = req.params['id'];
  if (!accountId) {
    throw new ValidationError('Account ID is required');
  }

  const account = await accountingService.getAccount(accountId);
  if (!account) {
    throw new NotFoundError(`Account with ID ${req.params['id']} not found`);
  }

  res.status(StatusCodes.OK).json({
    success: true,
    data: account,
  });
});

/**
 * @swagger
 * /api/accounts:
 *   post:
 *     summary: Create a new account
 *     tags: [Accounts]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/Account'
 */
router.post(
  '/',
  validateRequest(accountSchema),
  async (req: Request, res: Response) => {
    const account = await accountingService.createAccount(req.body);
    res.status(StatusCodes.CREATED).json({
      success: true,
      data: account,
    });
  }
);

/**
 * @swagger
 * /api/accounts/{id}:
 *   put:
 *     summary: Update an account
 *     tags: [Accounts]
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 */
router.put(
  '/:id',
  validateRequest(accountSchema),
  async (req: Request, res: Response) => {
    const accountId = req.params['id'];
    if (!accountId) {
      throw new ValidationError('Account ID is required');
    }

    const account = await accountingService.updateAccount(accountId, req.body);
    if (!account) {
      throw new NotFoundError(`Account with ID ${req.params['id']} not found`);
    }

    res.status(StatusCodes.OK).json({
      success: true,
      data: account,
    });
  }
);

/**
 * @swagger
 * /api/accounts/{id}:
 *   delete:
 *     summary: Delete an account
 *     tags: [Accounts]
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 */
router.delete('/:id', async (req: Request, res: Response) => {
  const accountId = req.params['id'];
  if (!accountId) {
    throw new ValidationError('Account ID is required');
  }

  const transactions = await accountingService.getTransactions({ accountId });
  const hasTransactions = transactions.length > 0;
  if (hasTransactions) {
    throw new ValidationError(
      'Cannot delete account with existing transactions'
    );
  }

  await accountingService.deleteAccount(accountId);
  res.status(StatusCodes.NO_CONTENT).send();
});

/**
 * @swagger
 * /api/accounts/{id}/balance:
 *   get:
 *     summary: Get account balance
 *     tags: [Accounts]
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *       - in: query
 *         name: asOf
 *         schema:
 *           type: string
 *           format: date
 */
router.get('/:id/balance', async (req: Request, res: Response) => {
  const accountId = req.params['id'];
  if (!accountId) {
    throw new ValidationError('Account ID is required');
  }

  const asOf = req.query['asOf'] ? new Date(req.query['asOf'] as string) : new Date();
  
  const account = await accountingService.getAccount(accountId);
  if (!account) {
    throw new NotFoundError(`Account with ID ${accountId} not found`);
  }

  const transactions = await accountingService.getTransactions({ accountId, asOf });
  const balance = transactions.reduce((sum, t) => sum + t.amount, 0);
  
  res.status(StatusCodes.OK).json({
    success: true,
    data: {
      accountId,
      balance,
      asOf: asOf.toISOString(),
    },
  });
});

export const accountRoutes = router;
