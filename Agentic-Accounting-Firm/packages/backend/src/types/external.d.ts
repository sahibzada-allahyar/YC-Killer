declare module 'intuit-oauth' {
  interface OAuthClientConfig {
    clientId: string;
    clientSecret: string;
    environment: string;
    redirectUri: string;
  }

  interface OAuthToken {
    token: {
      refresh_token: string;
    };
  }

  export default class OAuthClient {
    constructor(config: OAuthClientConfig);
    refreshUsingToken(refreshToken: string): Promise<OAuthToken>;
  }
}

declare module 'node-quickbooks' {
  interface QBConfig {
    token?: string;
    realmId?: string;
    useSandbox?: boolean;
    debug?: boolean;
    minorversion?: string;
  }

  type Callback<T> = (err: Error | null, entity?: T) => void;

  export default class QuickBooks {
    constructor(
      clientId: string,
      clientSecret: string,
      token: string,
      useTokenSecret: boolean,
      environment: string,
      debug: boolean,
      minorversion: string | null,
      oauthversion: string,
      refreshToken: string
    );

    // Account operations
    findAccounts(query: object, callback: Callback<QBOAccount[]>): void;
    getAccount(id: string, callback: Callback<QBOAccount>): void;
    createAccount(entity: object, callback: Callback<QBOAccount>): void;
    updateAccount(entity: object, callback: Callback<QBOAccount>): void;
    deleteAccount(id: string, callback: Callback<void>): void;

    // Transaction operations
    findTransactions(query: object, callback: Callback<QBOTransaction[]>): void;
    getTransaction(id: string, callback: Callback<QBOTransaction>): void;
    createTransaction(entity: object, callback: Callback<QBOTransaction>): void;
    updateTransaction(entity: object, callback: Callback<QBOTransaction>): void;
    deleteTransaction(id: string, callback: Callback<void>): void;

    // Journal entry operations
    findJournalEntries(query: object, callback: Callback<QBOJournalEntry[]>): void;
    createJournalEntry(entity: object, callback: Callback<QBOJournalEntry>): void;

    // Budget operations
    findBudgets(query: object, callback: Callback<QBOBudget[]>): void;
    createBudget(entity: object, callback: Callback<QBOBudget>): void;
    updateBudget(entity: object, callback: Callback<QBOBudget>): void;

    // Tax operations
    findTaxRates(query: object, callback: Callback<QBOTaxRate[]>): void;

    // Report operations
    report(type: string, params: object, callback: Callback<QBOReport>): void;
  }

  interface QBORef {
    value: string;
    name?: string;
  }

  interface QBOMetadata {
    CreateTime: string;
    LastUpdatedTime: string;
  }

  interface QBOAccount {
    Id: string;
    Name: string;
    AccountType: string;
    AccountSubType?: string;
    AcctNum?: string;
    CurrentBalance?: number;
    CurrencyRef?: QBORef;
    ParentRef?: QBORef;
    Description?: string;
    Active?: boolean;
    MetaData: QBOMetadata;
  }

  interface QBOTransaction {
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

  interface QBOJournalEntryLine {
    DetailType: string;
    Amount: number;
    JournalEntryLineDetail: {
      PostingType: string;
      AccountRef: QBORef;
      Description?: string;
    };
  }

  interface QBOJournalEntry {
    Id: string;
    TxnDate: string;
    DocNumber?: string;
    PrivateNote?: string;
    Line: QBOJournalEntryLine[];
    status: string;
    MetaData: QBOMetadata;
  }

  interface QBOBudgetDetail {
    AccountRef: QBORef;
    Amount: number;
    ActualAmount?: number;
  }

  interface QBOBudget {
    Id: string;
    Name: string;
    StartDate: string;
    EndDate: string;
    BudgetType: string;
    Active: boolean;
    BudgetDetail: QBOBudgetDetail[];
    MetaData: QBOMetadata;
  }

  interface QBOTaxRate {
    Id: string;
    Name: string;
    Description?: string;
    RateValue: number;
    AgencyRef?: QBORef;
    TaxCode?: string;
    Active?: boolean;
    EffectiveDate: string;
    MetaData: QBOMetadata;
  }

  interface QBOReport {
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
}
