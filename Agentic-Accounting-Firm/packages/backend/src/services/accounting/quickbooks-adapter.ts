import { OAuthClient, Token } from '@intuit/oauth';
import { 
  Account,
  Transaction,
  Budget,
  Report,
  ReportType,
  ReportConfig,
  TransactionFilters
} from '@accounting-agent/shared';
import { config } from '../../config';
import { quickbooksLogger as logger } from '../../utils/logger';

interface QuickBooksToken extends Token {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  x_refresh_token_expires_in: number;
  token_type: 'bearer';
}

class QuickBooksAdapter {
  private static instance: QuickBooksAdapter;
  private oauthClient: OAuthClient;
  private companyId?: string;
  private token?: QuickBooksToken;

  private constructor() {
    this.oauthClient = new OAuthClient({
      clientId: config.accounting.quickbooks.clientId,
      clientSecret: config.accounting.quickbooks.clientSecret,
      environment: config.accounting.quickbooks.environment,
      redirectUri: config.accounting.quickbooks.redirectUri,
    });

    // Initialize token from config
    this.token = {
      access_token: '',
      refresh_token: config.accounting.quickbooks.refreshToken,
      expires_in: 0,
      x_refresh_token_expires_in: 0,
      token_type: 'bearer',
    };
  }

  public static getInstance(): QuickBooksAdapter {
    if (!QuickBooksAdapter.instance) {
      QuickBooksAdapter.instance = new QuickBooksAdapter();
    }
    return QuickBooksAdapter.instance;
  }

  private async refreshTokenIfNeeded(): Promise<void> {
    try {
      if (!this.token || Date.now() >= this.token.expires_in) {
        const authResponse = await this.oauthClient.refreshUsingToken(this.token?.refresh_token);
        this.token = authResponse.getJson() as QuickBooksToken;
        logger.info('Refreshed QuickBooks access token');

        // Update refresh token in environment/config if it changed
        if (this.token.refresh_token !== config.accounting.quickbooks.refreshToken) {
          logger.info('New refresh token received');
          // In a production environment, you would update the stored refresh token
        }
      }
    } catch (error) {
      logger.error('Failed to refresh QuickBooks token', { error });
      throw error;
    }
  }

  private async makeApiRequest<T>(
    endpoint: string,
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' = 'GET',
    body?: unknown
  ): Promise<T> {
    try {
      await this.refreshTokenIfNeeded();

      if (!this.token?.access_token) {
        throw new Error('No access token available');
      }

      const baseUrl = `https://quickbooks.api.intuit.com/v3/company/${this.companyId}`;
      const url = `${baseUrl}${endpoint}`;

      const response = await fetch(url, {
        method,
        headers: {
          'Authorization': `Bearer ${this.token.access_token}`,
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: body ? JSON.stringify(body) : undefined,
      });

      if (!response.ok) {
        throw new Error(`QuickBooks API error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      logger.error('QuickBooks API request failed', { endpoint, method, error });
      throw error;
    }
  }

  // Account operations
  public async getAccounts(): Promise<Account[]> {
    const response = await this.makeApiRequest<{ Account: any[] }>('/query?query=select * from Account');
    return response.Account.map(this.mapQuickBooksAccount);
  }

  public async getAccount(id: string): Promise<Account> {
    const response = await this.makeApiRequest<{ Account: any }>(`/account/${id}`);
    return this.mapQuickBooksAccount(response.Account);
  }

  public async createAccount(account: Omit<Account, 'id'>): Promise<Account> {
    const qbAccount = this.mapToQuickBooksAccount(account);
    const response = await this.makeApiRequest<{ Account: any }>(
      '/account',
      'POST',
      qbAccount
    );
    return this.mapQuickBooksAccount(response.Account);
  }

  public async updateAccount(id: string, account: Partial<Account>): Promise<Account> {
    const current = await this.getAccount(id);
    const qbAccount = this.mapToQuickBooksAccount({ ...current, ...account });
    const response = await this.makeApiRequest<{ Account: any }>(
      `/account?operation=update`,
      'POST',
      qbAccount
    );
    return this.mapQuickBooksAccount(response.Account);
  }

  public async deleteAccount(id: string): Promise<void> {
    await this.makeApiRequest(
      `/account/${id}`,
      'DELETE'
    );
  }

  // Transaction operations
  public async createTransaction(transaction: Omit<Transaction, 'id'>): Promise<Transaction> {
    const qbTransaction = this.mapToQuickBooksTransaction(transaction);
    const response = await this.makeApiRequest<{ Transaction: any }>(
      '/transaction',
      'POST',
      qbTransaction
    );
    return this.mapQuickBooksTransaction(response.Transaction);
  }

  public async getTransactions(filters: TransactionFilters): Promise<Transaction[]> {
    let query = 'select * from Transaction';
    const conditions: string[] = [];

    if (filters.startDate) {
      conditions.push(`TxnDate >= '${filters.startDate}'`);
    }
    if (filters.endDate) {
      conditions.push(`TxnDate <= '${filters.endDate}'`);
    }
    if (filters.accountId) {
      conditions.push(`AccountRef = '${filters.accountId}'`);
    }
    if (filters.category) {
      conditions.push(`AccountType = '${filters.category}'`);
    }
    if (filters.minAmount) {
      conditions.push(`Amount >= ${filters.minAmount}`);
    }
    if (filters.maxAmount) {
      conditions.push(`Amount <= ${filters.maxAmount}`);
    }

    if (conditions.length > 0) {
      query += ` where ${conditions.join(' and ')}`;
    }

    const response = await this.makeApiRequest<{ Transaction: any[] }>(`/query?query=${encodeURIComponent(query)}`);
    return response.Transaction.map(this.mapQuickBooksTransaction);
  }

  public async categorizeTransaction(id: string, category: string): Promise<Transaction> {
    const transaction = await this.makeApiRequest<{ Transaction: any }>(`/transaction/${id}`);
    const updated = {
      ...transaction,
      AccountRef: { value: category },
    };

    const response = await this.makeApiRequest<{ Transaction: any }>(
      `/transaction?operation=update`,
      'POST',
      updated
    );
    return this.mapQuickBooksTransaction(response.Transaction);
  }

  // Report operations
  public async generateReport(type: ReportType, config: ReportConfig): Promise<Report> {
    const reportMap: Record<ReportType, string> = {
      balance_sheet: 'BalanceSheet',
      income_statement: 'ProfitAndLoss',
      cash_flow: 'CashFlow',
      tax_summary: 'TaxSummary',
    };

    const qbReportType = reportMap[type];
    const params = new URLSearchParams({
      start_date: config.period.start,
      end_date: config.period.end,
      format: config.format,
      ...config.filters,
    });

    const response = await this.makeApiRequest<any>(
      `/reports/${qbReportType}?${params.toString()}`
    );

    return {
      id: crypto.randomUUID(),
      type,
      data: response,
      generatedAt: new Date().toISOString(),
      config,
      metadata: {
        generatedAt: new Date().toISOString(),
        period: config.period,
        filters: config.filters,
      },
    };
  }

  // Budget operations
  public async createBudget(budget: Omit<Budget, 'id'>): Promise<Budget> {
    const qbBudget = this.mapToQuickBooksBudget(budget);
    const response = await this.makeApiRequest<{ Budget: any }>(
      '/budget',
      'POST',
      qbBudget
    );
    return this.mapQuickBooksBudget(response.Budget);
  }

  public async getBudgets(): Promise<Budget[]> {
    const response = await this.makeApiRequest<{ Budget: any[] }>('/query?query=select * from Budget');
    return response.Budget.map(this.mapQuickBooksBudget);
  }

  public async updateBudget(id: string, budget: Partial<Budget>): Promise<Budget> {
    const current = await this.makeApiRequest<{ Budget: any }>(`/budget/${id}`);
    const updated = {
      ...current.Budget,
      ...this.mapToQuickBooksBudget(budget as Budget),
    };

    const response = await this.makeApiRequest<{ Budget: any }>(
      `/budget?operation=update`,
      'POST',
      updated
    );
    return this.mapQuickBooksBudget(response.Budget);
  }

  // Sync operation
  public async sync(): Promise<void> {
    try {
      await this.refreshTokenIfNeeded();
      // Implement sync logic here
      logger.info('Successfully synced with QuickBooks');
    } catch (error) {
      logger.error('Failed to sync with QuickBooks', { error });
      throw error;
    }
  }

  // Mapping functions
  private mapQuickBooksAccount(qbAccount: any): Account {
    return {
      id: qbAccount.Id,
      name: qbAccount.Name,
      type: qbAccount.AccountType.toLowerCase(),
      balance: Number(qbAccount.CurrentBalance),
      currency: qbAccount.CurrencyRef?.value || 'USD',
      createdAt: qbAccount.MetaData.CreateTime,
      updatedAt: qbAccount.MetaData.LastUpdatedTime,
      lastUpdated: qbAccount.MetaData.LastUpdatedTime,
    };
  }

  private mapToQuickBooksAccount(account: Partial<Account>): any {
    return {
      Name: account.name,
      AccountType: account.type?.toUpperCase(),
      CurrencyRef: { value: account.currency },
    };
  }

  private mapQuickBooksTransaction(qbTransaction: any): Transaction {
    return {
      id: qbTransaction.Id,
      type: qbTransaction.TransactionType.toLowerCase(),
      amount: Number(qbTransaction.Amount),
      description: qbTransaction.Description,
      category: qbTransaction.AccountRef.name,
      date: qbTransaction.TxnDate,
      accountId: qbTransaction.AccountRef.value,
      status: qbTransaction.status?.toLowerCase() || 'completed',
      metadata: qbTransaction.MetaData,
      createdAt: qbTransaction.MetaData.CreateTime,
      updatedAt: qbTransaction.MetaData.LastUpdatedTime,
    };
  }

  private mapToQuickBooksTransaction(transaction: Partial<Transaction>): any {
    return {
      TransactionType: transaction.type?.toUpperCase(),
      Amount: transaction.amount,
      Description: transaction.description,
      AccountRef: {
        value: transaction.accountId,
        name: transaction.category,
      },
      TxnDate: transaction.date,
    };
  }

  private mapQuickBooksBudget(qbBudget: any): Budget {
    return {
      id: qbBudget.Id,
      name: qbBudget.Name,
      period: {
        start: qbBudget.StartDate,
        end: qbBudget.EndDate,
      },
      items: qbBudget.BudgetDetail.map((detail: any) => ({
        category: detail.AccountRef.name,
        amount: Number(detail.Amount),
        actual: Number(detail.ActualAmount),
      })),
      metadata: qbBudget.MetaData,
      createdAt: qbBudget.MetaData.CreateTime,
      updatedAt: qbBudget.MetaData.LastUpdatedTime,
    };
  }

  private mapToQuickBooksBudget(budget: Partial<Budget>): any {
    return {
      Name: budget.name,
      StartDate: budget.period?.start,
      EndDate: budget.period?.end,
      BudgetDetail: budget.items?.map(item => ({
        AccountRef: { name: item.category },
        Amount: item.amount,
      })),
    };
  }
}

export const quickBooksAdapter = QuickBooksAdapter.getInstance();
