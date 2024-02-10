"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const RoutesBase_1 = __importDefault(require("./RoutesBase"));
class Routes extends RoutesBase_1.default {
    /**
     * @deprecated - Use v1.phoneNumbers instead
     */
    get phoneNumbers() {
        console.warn("phoneNumbers is deprecated. Use v1.phoneNumbers instead.");
        return this.v2.phoneNumbers;
    }
    /**
     * @deprecated - Use v1.sipDomains instead
     */
    get sipDomains() {
        console.warn("sipDomains is deprecated. Use v1.sipDomains instead.");
        return this.v2.sipDomains;
    }
    /**
     * @deprecated - Use v1.trunks instead
     */
    get trunks() {
        console.warn("trunks is deprecated. Use v1.trunks instead.");
        return this.v2.trunks;
    }
}
module.exports = Routes;
