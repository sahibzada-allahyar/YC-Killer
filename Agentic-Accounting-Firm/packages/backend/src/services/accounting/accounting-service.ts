import { 
  CommandIntent,
  CommandEntity,
  Transaction,
  Account,
  Report,
  ReportType,
  Budget,
  TransactionFilters
} from '@accounting-agent/shared';
import { quickBooksAdapter } from './quickbooks-adapter';
import { accountingLogger as logger } from '../../utils/logger';

class AccountingService {
  private static instance: AccountingService;

  private constructor() {}

  public static getInstance(): AccountingService {
    if (!AccountingService.instance) {
      AccountingService.instance = new AccountingService();
    }
    return AccountingService.instance;
  }

  public async executeCommand(command: {
    intent: CommandIntent;
    entities: Record<string, CommandEntity>;
  }): Promise<unknown> {
    try {
      logger.info('Executing accounting command', { command });

      switch (command.intent) {
        case 'create_transaction':
          return this.createTransaction(command.entities);

        case 'get_transactions':
          return this.getTransactions(command.entities);

        case 'generate_report':
          return this.generateReport(command.entities);

        case 'get_account_balance':
          return this.getAccountBalance(command.entities);

        case 'categorize_transaction':
          return this.categorizeTransaction(command.entities);

        default:
          throw new Error(`Unknown command intent: ${command.intent}`);
      }
    } catch (error) {
      logger.error('Failed to execute command', { command, error });
      throw error;
    }
  }

  // Account operations
  public async getAccounts(): Promise<Account[]> {
    try {
      return await quickBooksAdapter.getAccounts();
    } catch (error) {
      logger.error('Failed to get accounts', { error });
      throw error;
    }
  }

  public async getAccount(id: string): Promise<Account> {
    try {
      return await quickBooksAdapter.getAccount(id);
    } catch (error) {
      logger.error('Failed to get account', { id, error });
      throw error;
    }
  }

  public async createAccount(accountData: Omit<Account, 'id' | 'createdAt' | 'updatedAt' | 'lastUpdated'>): Promise<Account> {
    try {
      const now = new Date().toISOString();
      const account: Omit<Account, 'id'> = {
        ...accountData,
        createdAt: now,
        updatedAt: now,
        lastUpdated: now,
      };
      return await quickBooksAdapter.createAccount(account);
    } catch (error) {
      logger.error('Failed to create account', { accountData, error });
      throw error;
    }
  }

  public async updateAccount(id: string, account: Partial<Account>): Promise<Account> {
    try {
      return await quickBooksAdapter.updateAccount(id, account);
    } catch (error) {
      logger.error('Failed to update account', { id, account, error });
      throw error;
    }
  }

  public async deleteAccount(id: string): Promise<void> {
    try {
      await quickBooksAdapter.deleteAccount(id);
    } catch (error) {
      logger.error('Failed to delete account', { id, error });
      throw error;
    }
  }

  // Transaction operations
  public async createTransaction(entities: Record<string, CommandEntity>): Promise<Transaction> {
    try {
      const now = new Date().toISOString();
      const transaction: Omit<Transaction, 'id'> = {
        type: entities['type'] as Transaction['type'],
        amount: Number(entities['amount']),
        description: String(entities['description'] || ''),
        category: String(entities['category'] || 'Uncategorized'),
        date: String(entities['date'] || now.split('T')[0]),
        accountId: String(entities['accountId']),
        status: 'pending' as const,
        metadata: entities['metadata'] ? 
          JSON.parse(String(entities['metadata'])) : 
          undefined,
        createdAt: now,
        updatedAt: now,
      };

      return await quickBooksAdapter.createTransaction(transaction);
    } catch (error) {
      logger.error('Failed to create transaction', { entities, error });
      throw error;
    }
  }

  public async getTransactions(entities: Record<string, CommandEntity>): Promise<Transaction[]> {
    try {
      const filters: TransactionFilters = {
        startDate: entities['startDate'] as string,
        endDate: entities['endDate'] as string,
        accountId: entities['accountId'] as string,
        category: entities['category'] as string,
        minAmount: entities['minAmount'] ? Number(entities['minAmount']) : undefined,
        maxAmount: entities['maxAmount'] ? Number(entities['maxAmount']) : undefined,
      };

      return await quickBooksAdapter.getTransactions(filters);
    } catch (error) {
      logger.error('Failed to get transactions', { entities, error });
      throw error;
    }
  }

  private async generateReport(entities: Record<string, CommandEntity>): Promise<Report> {
    try {
      const reportType = entities['type'] as ReportType;
      const config = {
        type: reportType,
        period: {
          start: String(entities['startDate'] || new Date().toISOString().split('T')[0]),
          end: String(entities['endDate'] || new Date().toISOString().split('T')[0]),
        },
        format: 'json' as const,
        filters: entities['filters'] ? 
          JSON.parse(String(entities['filters'])) : 
          {},
      };

      return await quickBooksAdapter.generateReport(reportType, config);
    } catch (error) {
      logger.error('Failed to generate report', { entities, error });
      throw error;
    }
  }

  private async getAccountBalance(entities: Record<string, CommandEntity>): Promise<Account> {
    try {
      const accountId = String(entities['accountId']);
      return await quickBooksAdapter.getAccount(accountId);
    } catch (error) {
      logger.error('Failed to get account balance', { entities, error });
      throw error;
    }
  }

  private async categorizeTransaction(entities: Record<string, CommandEntity>): Promise<Transaction> {
    try {
      const transactionId = String(entities['transactionId']);
      const category = String(entities['category']);
      return await quickBooksAdapter.categorizeTransaction(transactionId, category);
    } catch (error) {
      logger.error('Failed to categorize transaction', { entities, error });
      throw error;
    }
  }

  // Budget operations
  public async createBudget(budgetData: Omit<Budget, 'id' | 'createdAt' | 'updatedAt'>): Promise<Budget> {
    try {
      const now = new Date().toISOString();
      const budget: Omit<Budget, 'id'> = {
        ...budgetData,
        createdAt: now,
        updatedAt: now,
      };
      return await quickBooksAdapter.createBudget(budget);
    } catch (error) {
      logger.error('Failed to create budget', { budgetData, error });
      throw error;
    }
  }

  public async getBudgets(): Promise<Budget[]> {
    try {
      return await quickBooksAdapter.getBudgets();
    } catch (error) {
      logger.error('Failed to get budgets', { error });
      throw error;
    }
  }

  public async updateBudget(id: string, budget: Partial<Budget>): Promise<Budget> {
    try {
      return await quickBooksAdapter.updateBudget(id, budget);
    } catch (error) {
      logger.error('Failed to update budget', { id, budget, error });
      throw error;
    }
  }

  // Sync operations
  public async syncAccounts(): Promise<void> {
    try {
      await quickBooksAdapter.sync();
    } catch (error) {
      logger.error('Failed to sync accounts', { error });
      throw error;
    }
  }
}

export const accountingService = AccountingService.getInstance();
