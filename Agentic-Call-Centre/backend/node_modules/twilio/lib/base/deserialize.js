"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.integer = exports.decimal = exports.rfc2822DateTime = exports.iso8601DateTime = exports.iso8601Date = void 0;
const dayjs_1 = __importDefault(require("dayjs"));
const utc_1 = __importDefault(require("dayjs/plugin/utc"));
dayjs_1.default.extend(utc_1.default);
/**
 * @namespace deserialize
 */
/**
 * Parse a string into a Date object
 *
 * @param s - Date string in YYYY-MM-DD format
 * @returns Date object, or the original date string if the argument is not a valid date
 */
function iso8601Date(s) {
    return parseDate(s, "YYYY-MM-DD");
}
exports.iso8601Date = iso8601Date;
/**
 * Parse a string into a Date object
 *
 * @param s - Date string in YYYY-MM-DD[T]HH:mm:ss[Z] format
 * @returns Date object, or the original date string if the argument is not a valid date
 */
function iso8601DateTime(s) {
    return parseDate(s, "YYYY-MM-DD[T]HH:mm:ss[Z]");
}
exports.iso8601DateTime = iso8601DateTime;
/**
 * Parse a string into a Date object
 *
 * @param s - Date string in ddd, DD MMM YYYY HH:mm:ss [+0000] format
 * @returns Date object, or the original date string if the argument is not a valid date
 */
function rfc2822DateTime(s) {
    return parseDate(s, "ddd, DD MMM YYYY HH:mm:ss [+0000]");
}
exports.rfc2822DateTime = rfc2822DateTime;
/**
 * Parse a string into a decimal
 *
 * @param d - Decimal value as string
 * @returns Number, or the original string if the argument is NaN
 */
function decimal(d) {
    return parseNumber(d, parseFloat);
}
exports.decimal = decimal;
/**
 * Parse a string into a integer
 *
 * @param i - Integer value as string
 * @returns Number, or the original string if the argument is NaN
 */
function integer(i) {
    return parseNumber(i, parseInt);
}
exports.integer = integer;
function parseDate(s, format) {
    var m = dayjs_1.default.utc(s, format);
    if (m.isValid()) {
        return m.toDate();
    }
    return s;
}
function parseNumber(n, parser) {
    var parsed = parser(n);
    if (typeof parsed === "number" && isNaN(parsed)) {
        return n;
    }
    return parsed;
}
