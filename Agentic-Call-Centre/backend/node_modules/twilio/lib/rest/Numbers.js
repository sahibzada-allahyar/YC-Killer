"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const NumbersBase_1 = __importDefault(require("./NumbersBase"));
class Numbers extends NumbersBase_1.default {
    /**
     * @deprecated - Use v2.regulatoryCompliance instead
     */
    get regulatoryCompliance() {
        console.warn("regulatoryCompliance is deprecated. Use v2.regulatoryCompliance instead.");
        return this.v2.regulatoryCompliance;
    }
}
module.exports = Numbers;
