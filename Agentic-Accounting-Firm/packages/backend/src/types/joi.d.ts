declare module 'joi' {
  export interface ValidationError extends Error {
    details: Array<{
      message: string;
      path: Array<string | number>;
      type: string;
      context?: Record<string, any>;
    }>;
    _original: any;
    annotate(): string;
  }

  export interface ValidationResult<T> {
    error?: ValidationError;
    value: T;
    warning?: ValidationError;
  }

  export interface ValidationOptions {
    abortEarly?: boolean;
    allowUnknown?: boolean;
    context?: object;
    convert?: boolean;
    dateFormat?: 'date' | 'iso' | 'string' | 'time' | 'utc';
    debug?: boolean;
    errors?: {
      escapeHtml?: boolean;
      language?: string;
      wrap?: {
        label?: string;
        array?: string;
      };
    };
    externals?: boolean;
    messages?: { [key: string]: string };
    noDefaults?: boolean;
    nonEnumerables?: boolean;
    presence?: 'required' | 'optional' | 'forbidden';
    skipFunctions?: boolean;
    stripUnknown?: boolean | { arrays?: boolean; objects?: boolean };
  }

  export class ValidationErrorItem {
    message: string;
    path: Array<string | number>;
    type: string;
    context?: Record<string, any>;
  }

  export interface Schema {
    validate(value: any, options?: ValidationOptions): ValidationResult<any>;
  }

  export interface StringSchema extends Schema {
    min(limit: number): this;
    max(limit: number): this;
    length(limit: number): this;
    email(): this;
    required(): this;
    optional(): this;
    pattern(regex: RegExp): this;
    valid(...values: any[]): this;
    messages(messages: { [key: string]: string }): this;
  }

  export interface NumberSchema extends Schema {
    min(limit: number | Reference): this;
    max(limit: number | Reference): this;
    required(): this;
    optional(): this;
    default(value: number): this;
    messages(messages: { [key: string]: string }): this;
  }

  export interface DateSchema extends Schema {
    min(date: Date | string | Reference): this;
    max(date: Date | string | Reference): this;
    required(): this;
    optional(): this;
    default(value: Date | string): this;
    messages(messages: { [key: string]: string }): this;
  }

  export type Reference = {
    [Symbol.toStringTag]: 'Reference';
  } & {
    ref: string;
  };

  export interface ObjectSchema extends Schema {
    keys(schema: { [key: string]: Schema }): this;
    required(): this;
    optional(): this;
  }

  export interface ArraySchema extends Schema {
    items(type: Schema): this;
    min(limit: number): this;
    max(limit: number): this;
    required(): this;
    optional(): this;
  }

  export function string(): StringSchema;
  export function number(): NumberSchema;
  export function object(schema?: { [key: string]: Schema }): ObjectSchema;
  export function array(): ArraySchema;
  export function date(): DateSchema;
  export function ref(path: string): Reference & { ref: string };
  export interface BinarySchema extends Schema {
    required(): this;
    optional(): this;
    min(limit: number): this;
    max(limit: number): this;
    encoding(encoding: string): this;
  }

  export function binary(): BinarySchema;
  export function validate(value: any, schema: Schema, options?: ValidationOptions): ValidationResult<any>;
}
