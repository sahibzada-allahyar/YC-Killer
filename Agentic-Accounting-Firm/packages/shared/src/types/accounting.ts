export interface Account {
  id: string;
  name: string;
  type: 'asset' | 'liability' | 'equity' | 'revenue' | 'expense';
  description?: string;
  currency: string;
  parentAccountId?: string;
  balance?: number;
  status?: 'active' | 'inactive' | 'archived';
  lastUpdated?: string;
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, unknown>;
}

export interface Transaction {
  id: string;
  date: string;
  description: string;
  amount: number;
  type: 'debit' | 'credit';
  accountId: string;
  category: string;
  reference?: string;
  status?: 'pending' | 'completed' | 'failed' | 'cancelled';
  metadata?: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface TransactionQuery {
  startDate?: string;
  endDate?: string;
  accountId?: string;
  category?: string;
  type?: 'debit' | 'credit';
  minAmount?: number;
  maxAmount?: number;
  limit?: number;
  offset?: number;
}

export interface TransactionCreateInput {
  date: string | Date;
  description: string;
  amount: number;
  type: 'debit' | 'credit';
  accountId: string;
  category: string;
  reference?: string;
  metadata?: Record<string, unknown>;
}

export interface TransactionUpdateInput extends Partial<TransactionCreateInput> {
  id: string;
}

export interface TransactionCategorization {
  transactionId: string;
  suggestedCategory: string;
  confidence: number;
}

export interface ChartOfAccount {
  id: string;
  name: string;
  accounts: Account[];
  createdAt: string;
  updatedAt: string;
}

export interface JournalEntry {
  id: string;
  date: string;
  description: string;
  entries: Array<{
    accountId: string;
    type: 'debit' | 'credit';
    amount: number;
  }>;
  status: 'draft' | 'posted' | 'cancelled';
  createdAt: string;
  updatedAt: string;
}

export interface BudgetItem {
  accountId: string;
  amount: number;
  type: 'income' | 'expense';
  category: string;
}

export interface Budget {
  id: string;
  name: string;
  period: {
    start: string;
    end: string;
  };
  type: 'monthly' | 'quarterly' | 'yearly';
  items: BudgetItem[];
  createdAt: string;
  updatedAt: string;
}

export type ReportType = 
  | 'income_statement'
  | 'balance_sheet'
  | 'cash_flow'
  | 'trial_balance'
  | 'accounts_receivable'
  | 'accounts_payable'
  | 'tax_summary';

export interface DateRange {
  start: string;
  end: string;
}

export interface ReportConfig {
  type: ReportType;
  period: DateRange;
  format?: 'json' | 'csv' | 'pdf';
  groupBy?: 'month' | 'quarter' | 'year';
  filters?: Record<string, unknown>;
}

export interface Report {
  id: string;
  type: ReportType;
  name: string;
  data: Record<string, number>;
  period: DateRange;
  generatedAt: string;
  filters?: Record<string, unknown>;
}

export interface TaxRate {
  id: string;
  name: string;
  rate: number;
  type: 'sales' | 'income' | 'vat' | 'other';
  country: string;
  region?: string;
  effectiveDate: string;
  expiryDate?: string;
  createdAt: string;
  updatedAt: string;
}

export interface TransactionFilters {
  startDate?: string;
  endDate?: string;
  accountId?: string;
  category?: string;
  type?: 'debit' | 'credit';
  minAmount?: number;
  maxAmount?: number;
  status?: 'pending' | 'completed' | 'failed' | 'cancelled';
  search?: string;
}

export interface AccountingService {
  // Account methods
  getAccounts(): Promise<Account[]>;
  getAccount(id: string): Promise<Account | null>;
  createAccount(data: Omit<Account, 'id' | 'createdAt' | 'updatedAt'>): Promise<Account>;
  updateAccount(id: string, data: Partial<Account>): Promise<Account | null>;
  deleteAccount(id: string): Promise<void>;

  // Transaction methods
  getTransactions(query: TransactionQuery): Promise<Transaction[]>;
  getTransaction(id: string): Promise<Transaction | null>;
  createTransaction(data: TransactionCreateInput): Promise<Transaction>;
  updateTransaction(id: string, data: Partial<TransactionCreateInput>): Promise<Transaction | null>;
  deleteTransaction(id: string): Promise<void>;
  categorizeTransactions(transactionIds: string[]): Promise<TransactionCategorization[]>;

  // Balance methods
  getAccountBalance(accountId: string, asOf?: string): Promise<number>;
  getTrialBalance(asOf?: string): Promise<Record<string, number>>;

  // Reporting methods
  generateIncomeStatement(period: DateRange): Promise<Record<string, number>>;
  generateBalanceSheet(asOf: string): Promise<Record<string, number>>;
  generateCashFlow(period: DateRange): Promise<Record<string, number>>;
}

export interface AccountingAdapter {
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  isConnected(): boolean;
  refreshConnection(): Promise<void>;
  
  // Account operations
  fetchAccounts(): Promise<Account[]>;
  fetchAccount(id: string): Promise<Account | null>;
  createAccount(data: Omit<Account, 'id' | 'createdAt' | 'updatedAt'>): Promise<Account>;
  updateAccount(id: string, data: Partial<Account>): Promise<Account | null>;
  deleteAccount(id: string): Promise<void>;

  // Transaction operations
  fetchTransactions(query: TransactionQuery): Promise<Transaction[]>;
  fetchTransaction(id: string): Promise<Transaction | null>;
  createTransaction(data: TransactionCreateInput): Promise<Transaction>;
  updateTransaction(id: string, data: Partial<TransactionCreateInput>): Promise<Transaction | null>;
  deleteTransaction(id: string): Promise<void>;

  // Balance operations
  fetchAccountBalance(accountId: string, asOf?: string): Promise<number>;
  fetchTrialBalance(asOf?: string): Promise<Record<string, number>>;

  // Reporting operations
  fetchIncomeStatement(period: DateRange): Promise<Record<string, number>>;
  fetchBalanceSheet(asOf: string): Promise<Record<string, number>>;
  fetchCashFlow(period: DateRange): Promise<Record<string, number>>;
}
