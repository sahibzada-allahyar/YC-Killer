// Common metadata fields
interface BaseMetadata {
  createdBy?: string;
  updatedBy?: string;
  source?: string;
  tags?: string[];
  notes?: string;
}

// Transaction-specific metadata
export interface TransactionMetadata extends BaseMetadata {
  receiptUrl?: string;
  paymentMethod?: string;
  paymentStatus?: string;
  reconciled?: boolean;
  reconciledAt?: string;
  customFields?: Record<string, unknown>;
}

// Journal entry metadata
export interface JournalEntryMetadata extends BaseMetadata {
  approvedBy?: string;
  approvedAt?: string;
  documentRefs?: string[];
  adjustmentType?: string;
  period?: {
    year: number;
    month: number;
  };
}

// Budget metadata
export interface BudgetMetadata extends BaseMetadata {
  fiscalYear?: number;
  department?: string;
  version?: number;
  status?: 'draft' | 'approved' | 'active';
  approvedBy?: string;
  approvedAt?: string;
  assumptions?: string[];
}

// Tax rate metadata
export interface TaxRateMetadata extends BaseMetadata {
  jurisdiction?: string;
  category?: string;
  applicableTo?: string[];
  exemptions?: string[];
  lastReviewedAt?: string;
  lastReviewedBy?: string;
  regulatoryRef?: string;
}

// Report metadata
export interface ReportMetadata extends BaseMetadata {
  generatedBy?: string;
  format?: string;
  parameters?: Record<string, unknown>;
  filters?: Record<string, unknown>;
  customization?: {
    template?: string;
    branding?: boolean;
    sections?: string[];
  };
}

// Account metadata
export interface AccountMetadata extends BaseMetadata {
  accountNumber?: string;
  bankInfo?: {
    bankName?: string;
    branchCode?: string;
    swiftCode?: string;
  };
  restrictions?: string[];
  category?: string;
  subCategory?: string;
  department?: string;
  costCenter?: string;
}

// Vendor metadata
export interface VendorMetadata extends BaseMetadata {
  taxId?: string;
  paymentTerms?: string;
  creditLimit?: number;
  preferredPaymentMethod?: string;
  rating?: number;
  industry?: string;
  contacts?: {
    name: string;
    email: string;
    phone?: string;
    role?: string;
  }[];
}

// Customer metadata
export interface CustomerMetadata extends BaseMetadata {
  customerType?: string;
  segment?: string;
  creditRating?: string;
  paymentTerms?: string;
  taxExempt?: boolean;
  taxExemptionNumber?: string;
  contacts?: {
    name: string;
    email: string;
    phone?: string;
    role?: string;
  }[];
}

// Product/Service metadata
export interface ProductMetadata extends BaseMetadata {
  sku?: string;
  category?: string;
  subCategory?: string;
  unit?: string;
  taxCode?: string;
  customFields?: Record<string, unknown>;
}

// Document metadata
export interface DocumentMetadata extends BaseMetadata {
  documentType?: string;
  fileType?: string;
  fileSize?: number;
  uploadedBy?: string;
  uploadedAt?: string;
  expiresAt?: string;
  version?: number;
  status?: 'draft' | 'final' | 'archived';
}
