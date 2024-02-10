export interface NumberParser {
    (n: string): number;
}
/**
 * @namespace deserialize
 */
/**
 * Parse a string into a Date object
 *
 * @param s - Date string in YYYY-MM-DD format
 * @returns Date object, or the original date string if the argument is not a valid date
 */
export declare function iso8601Date(s: string): string | Date;
/**
 * Parse a string into a Date object
 *
 * @param s - Date string in YYYY-MM-DD[T]HH:mm:ss[Z] format
 * @returns Date object, or the original date string if the argument is not a valid date
 */
export declare function iso8601DateTime(s: string): string | Date;
/**
 * Parse a string into a Date object
 *
 * @param s - Date string in ddd, DD MMM YYYY HH:mm:ss [+0000] format
 * @returns Date object, or the original date string if the argument is not a valid date
 */
export declare function rfc2822DateTime(s: string): string | Date;
/**
 * Parse a string into a decimal
 *
 * @param d - Decimal value as string
 * @returns Number, or the original string if the argument is NaN
 */
export declare function decimal(d: string): string | number;
/**
 * Parse a string into a integer
 *
 * @param i - Integer value as string
 * @returns Number, or the original string if the argument is NaN
 */
export declare function integer(i: string): string | number;
