"use strict";
/**
 * @namespace values
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.of = void 0;
/**
 * Removes all undefined values of an object
 *
 * @param obj - object to filter
 * @returns object with no undefined values
 */
function of(obj) {
    return Object.fromEntries(Object.entries(obj).filter((entry) => entry[1] !== undefined));
}
exports.of = of;
