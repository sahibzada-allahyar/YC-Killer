"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const LookupsBase_1 = __importDefault(require("./LookupsBase"));
class Lookups extends LookupsBase_1.default {
    /**
     * @deprecated - Use v1.phoneNumbers instead
     */
    get phoneNumbers() {
        console.warn("phoneNumbers is deprecated. Use v1.phoneNumbers instead.");
        return this.v1.phoneNumbers;
    }
}
module.exports = Lookups;
