"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const PricingBase_1 = __importDefault(require("./PricingBase"));
class Pricing extends PricingBase_1.default {
    /**
     * @deprecated - Use v1.messaging instead
     */
    get messaging() {
        console.warn("messaging is deprecated. Use v1.messaging instead.");
        return this.v1.messaging;
    }
    /**
     * @deprecated - Use v1.phoneNumbers instead
     */
    get phoneNumbers() {
        console.warn("phoneNumbers is deprecated. Use v1.phoneNumbers instead.");
        return this.v1.phoneNumbers;
    }
    /**
     * @deprecated - Use v2.voice instead
     */
    get voice() {
        console.warn("voice is deprecated. Use v2.voice instead.");
        return this.v2.voice;
    }
    /**
     * @deprecated - Use v2.countries instead
     */
    get countries() {
        console.warn("countries is deprecated. Use v2.countries instead.");
        return this.v2.countries;
    }
    /**
     * @deprecated - Use v2.numbers instead
     */
    get numbers() {
        console.warn("numbers is deprecated. Use v2.numbers instead.");
        return this.v2.numbers;
    }
}
module.exports = Pricing;
