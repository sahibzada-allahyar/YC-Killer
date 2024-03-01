import { Router } from 'express';
import { z } from 'zod';
import { commandRoutes } from './command-routes';
import { accountRoutes } from './account-routes';
import { transactionRoutes } from './transaction-routes';

// Export all routes
export const routes = {
  commandRoutes,
  accountRoutes,
  transactionRoutes,
};

// Common validation schemas
export const commonSchemas = {
  // Pagination schema
  pagination: z.object({
    page: z.string().transform(Number).optional(),
    limit: z.string().transform(Number).optional(),
  }),

  // Date range schema
  dateRange: z.object({
    startDate: z.string().datetime(),
    endDate: z.string().datetime(),
  }),

  // ID parameter schema
  idParam: z.object({
    id: z.string().uuid(),
  }),

  // Query filters schema
  filters: z.object({
    search: z.string().optional(),
    status: z.string().optional(),
    type: z.string().optional(),
    category: z.string().optional(),
    minAmount: z.string().transform(Number).optional(),
    maxAmount: z.string().transform(Number).optional(),
  }),

  // Sort schema
  sort: z.object({
    field: z.string(),
    direction: z.enum(['asc', 'desc']),
  }).optional(),
};

// Command validation schemas
export const commandSchemas = {
  // Voice command schema
  voiceCommand: z.object({
    audio: z.any(), // Validated by express-fileupload
    metadata: z.record(z.unknown()).optional(),
  }),

  // Text command schema
  textCommand: z.object({
    text: z.string().min(1),
    context: z.object({
      userId: z.string().uuid(),
      timestamp: z.string().datetime(),
      source: z.enum(['voice', 'text', 'api', 'system']),
      metadata: z.record(z.unknown()).optional(),
      previousCommand: z.string().optional(),
      sessionId: z.string().uuid(),
    }),
  }),
};

// Account validation schemas
export const accountSchemas = {
  // Create account schema
  createAccount: z.object({
    name: z.string().min(1),
    type: z.enum(['checking', 'savings', 'credit', 'cash']),
    currency: z.string().length(3),
    metadata: z.record(z.unknown()).optional(),
  }),

  // Update account schema
  updateAccount: z.object({
    name: z.string().min(1).optional(),
    type: z.enum(['checking', 'savings', 'credit', 'cash']).optional(),
    metadata: z.record(z.unknown()).optional(),
  }),
};

// Transaction validation schemas
export const transactionSchemas = {
  // Create transaction schema
  createTransaction: z.object({
    type: z.enum(['income', 'expense', 'transfer']),
    amount: z.number().positive(),
    description: z.string().min(1),
    category: z.string().min(1),
    date: z.string().datetime(),
    accountId: z.string().uuid(),
    metadata: z.record(z.unknown()).optional(),
  }),

  // Update transaction schema
  updateTransaction: z.object({
    amount: z.number().positive().optional(),
    description: z.string().min(1).optional(),
    category: z.string().min(1).optional(),
    date: z.string().datetime().optional(),
    metadata: z.record(z.unknown()).optional(),
  }),
};

// Create router instance
const router = Router();

// Mount routes
router.use('/commands', commandRoutes);
router.use('/accounts', accountRoutes);
router.use('/transactions', transactionRoutes);

// Export router
export default router;
