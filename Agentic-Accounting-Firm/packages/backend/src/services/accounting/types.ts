import {
  Account,
  Transaction,
  ChartOfAccount,
  JournalEntry,
  Budget,
  ReportType,
  ReportConfig,
  TaxRate,
} from '@accounting-agent/shared';

// QuickBooks API Types
export interface QBOMetadata {
  CreateTime: string;
  LastUpdatedTime: string;
}

export interface QBORef {
  value: string;
  name?: string;
}

export interface QBOAccount {
  Id: string;
  Name: string;
  AccountType: string;
  AccountSubType?: string;
  CurrentBalance?: number;
  CurrencyRef?: QBORef;
  Active?: boolean;
  MetaData: QBOMetadata;
}

export interface QBOTransaction {
  Id: string;
  Type: string;
  Amount: number;
  Description?: string;
  CategoryRef?: QBORef;
  TxnDate: string;
  AccountRef: QBORef;
  Status: string;
  PaymentMethodRef?: QBORef;
  LocationRef?: QBORef;
  MetaData: QBOMetadata;
}

export interface QBOReport {
  Header: {
    Time: string;
    ReportName: string;
    StartPeriod: string;
    EndPeriod: string;
    Currency: string;
  };
  Rows: Array<{
    Header?: { [key: string]: string };
    Rows?: Array<{ ColData: Array<{ value: string }> }>;
    Summary?: { ColData: Array<{ value: string }> };
  }>;
}

// Adapter Types
export interface AuthCredentials {
  clientId: string;
  clientSecret: string;
  refreshToken: string;
  environment: 'sandbox' | 'production';
}

export interface TransactionFilters {
  startDate?: string;
  endDate?: string;
  accountId?: string;
  category?: string;
  minAmount?: number;
  maxAmount?: number;
  status?: string;
}

export interface JournalEntryFilters {
  startDate?: string;
  endDate?: string;
  status?: 'draft' | 'posted' | 'void';
}

export interface Report {
  id: string;
  type: ReportType;
  data: unknown;
  generatedAt: string;
  config: ReportConfig;
}

export interface SyncResult {
  success: boolean;
  timestamp: string;
  details: {
    accountsUpdated: number;
    transactionsUpdated: number;
    errors?: string[];
  };
}

// Adapter Interface
export interface AccountingSoftwareAdapter {
  // Account operations
  getAccounts(): Promise<Account[]>;
  getAccount(id: string): Promise<Account>;
  createAccount(account: Omit<Account, 'id'>): Promise<Account>;
  updateAccount(id: string, account: Partial<Account>): Promise<Account>;
  deleteAccount(id: string): Promise<void>;

  // Transaction operations
  getTransactions(filters?: TransactionFilters): Promise<Transaction[]>;
  getTransaction(id: string): Promise<Transaction>;
  createTransaction(transaction: Omit<Transaction, 'id'>): Promise<Transaction>;
  updateTransaction(id: string, transaction: Partial<Transaction>): Promise<Transaction>;
  deleteTransaction(id: string): Promise<void>;
  categorizeTransaction(id: string, category: string): Promise<Transaction>;

  // Chart of accounts operations
  getChartOfAccounts(): Promise<ChartOfAccount[]>;
  updateChartOfAccounts(accounts: ChartOfAccount[]): Promise<void>;

  // Journal entry operations
  createJournalEntry(entry: Omit<JournalEntry, 'id'>): Promise<JournalEntry>;
  getJournalEntries(filters?: JournalEntryFilters): Promise<JournalEntry[]>;

  // Budget operations
  getBudgets(): Promise<Budget[]>;
  createBudget(budget: Omit<Budget, 'id'>): Promise<Budget>;
  updateBudget(id: string, budget: Partial<Budget>): Promise<Budget>;

  // Tax operations
  getTaxRates(): Promise<TaxRate[]>;
  calculateTax(amount: number, taxRateId: string): Promise<number>;

  // Report operations
  generateReport(type: ReportType, config: ReportConfig): Promise<Report>;

  // Sync operations
  sync(): Promise<SyncResult>;
  getLastSyncTime(): Promise<string>;
}

// Error Types
export class AccountingSoftwareError extends Error {
  constructor(
    message: string,
    public code: string,
    public originalError?: unknown
  ) {
    super(message);
    this.name = 'AccountingSoftwareError';
  }
}
