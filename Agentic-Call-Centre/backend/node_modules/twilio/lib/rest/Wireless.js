"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const WirelessBase_1 = __importDefault(require("./WirelessBase"));
class Wireless extends WirelessBase_1.default {
    /**
     * @deprecated - Use v1.usageRecords instead
     */
    get usageRecords() {
        console.warn("usageRecords is deprecated. Use v1.usageRecords instead.");
        return this.v1.usageRecords;
    }
    /**
     * @deprecated - Use v1.commands instead
     */
    get commands() {
        console.warn("commands is deprecated. Use v1.commands instead.");
        return this.v1.commands;
    }
    /**
     * @deprecated - Use v1.ratePlans instead
     */
    get ratePlans() {
        console.warn("ratePlans is deprecated. Use v1.ratePlans instead.");
        return this.v1.ratePlans;
    }
    /**
     * @deprecated - Use v1.sims instead
     */
    get sims() {
        console.warn("sims is deprecated. Use v1.sims instead.");
        return this.v1.sims;
    }
}
module.exports = Wireless;
