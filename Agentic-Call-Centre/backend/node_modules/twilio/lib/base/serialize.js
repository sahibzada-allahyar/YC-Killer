"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.map = exports.twiml = exports.bool = exports.object = exports.prefixedCollapsibleMap = exports.iso8601DateTime = exports.iso8601Date = void 0;
const dayjs_1 = __importDefault(require("dayjs"));
const utc_1 = __importDefault(require("dayjs/plugin/utc"));
dayjs_1.default.extend(utc_1.default);
/**
 * @namespace serialize
 */
/**
 * Turns a Date object into a string if parameter is a Date otherwise returns the parameter
 *
 * @param d - date object to format
 * @returns date formatted in YYYY-MM-DD form, otherwise the
 * provided parameter.
 */
function iso8601Date(date) {
    if (!date || !(date instanceof Date)) {
        return date;
    }
    else {
        return dayjs_1.default.utc(date).format("YYYY-MM-DD");
    }
}
exports.iso8601Date = iso8601Date;
/**
 * Turns a Date object into a string if parameter is a Date otherwise returns the parameter
 *
 * @param  d - date object to format
 * @returns date formatted in YYYY-MM-DD[T]HH:mm:ss[Z] form, otherwise the
 * provided parameter.
 */
function iso8601DateTime(date) {
    if (!date || !(date instanceof Date)) {
        return date;
    }
    else {
        return dayjs_1.default.utc(date).format("YYYY-MM-DD[T]HH:mm:ss[Z]");
    }
}
exports.iso8601DateTime = iso8601DateTime;
function prefixedCollapsibleMap(m, prefix) {
    if (!m ||
        typeof m !== "object" ||
        Object.prototype.toString.call(m) !== "[object Object]") {
        return {};
    }
    function flatten(m, result, previous) {
        result = result || {};
        previous = previous || [];
        Object.keys(m).forEach((key) => {
            const unionKeys = [...previous];
            if (!unionKeys.includes(key)) {
                unionKeys.push(key);
            }
            if (typeof m[key] === "object" &&
                Object.prototype.toString.call(m[key]) === "[object Object]") {
                flatten(m[key], result, unionKeys);
            }
            else {
                result[unionKeys.join(".")] = m[key];
            }
        });
        return result;
    }
    var flattened = flatten(m);
    var result = flattened;
    if (prefix) {
        result = {};
        Object.keys(flattened).forEach((key) => {
            result[prefix + "." + key] = flattened[key];
        });
    }
    return result;
}
exports.prefixedCollapsibleMap = prefixedCollapsibleMap;
function object(o) {
    if (typeof o === "object") {
        return JSON.stringify(o);
    }
    return o;
}
exports.object = object;
/**
 * Coerces a boolean literal into a string
 *
 * @param input - boolean or string to be coerced
 * @returns a string 'true' or 'false' if passed a boolean, else the value
 */
function bool(input) {
    if (typeof input === "string") {
        return input;
    }
    if (typeof input === "boolean" ||
        (typeof input === "object" &&
            Object.prototype.toString.call(input) === "[object Boolean]")) {
        return input.toString();
    }
    return input;
}
exports.bool = bool;
function twiml(input) {
    return input.toString();
}
exports.twiml = twiml;
/**
 * Maps transform over each element in input if input is an array
 *
 * @param input - array to map transform over, if not an array then it is
 * returned as is.
 * @returns new array with transform applied to each element.
 */
function map(input, transform) {
    if (typeof input === "object" && Array.isArray(input)) {
        return input.map((element) => transform(element));
    }
    return input;
}
exports.map = map;
