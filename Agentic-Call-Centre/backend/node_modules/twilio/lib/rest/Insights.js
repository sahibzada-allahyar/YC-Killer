"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const InsightsBase_1 = __importDefault(require("./InsightsBase"));
class Insights extends InsightsBase_1.default {
    /**
     * @deprecated - Use v1.settings instead
     */
    get settings() {
        console.warn("settings is deprecated. Use v1.settings instead.");
        return this.v1.settings;
    }
    /**
     * @deprecated - Use v1.calls instead
     */
    get calls() {
        console.warn("calls is deprecated. Use v1.calls instead.");
        return this.v1.calls;
    }
    /**
     * @deprecated - Use v1.callSummaries instead
     */
    get callSummaries() {
        console.warn("callSummaries is deprecated. Use v1.callSummaries instead.");
        return this.v1.callSummaries;
    }
    /**
     * @deprecated - Use v1.conferences instead
     */
    get conferences() {
        console.warn("conferences is deprecated. Use v1.conferences instead.");
        return this.v1.conferences;
    }
    /**
     * @deprecated - Use v1.rooms instead
     */
    get rooms() {
        console.warn("rooms is deprecated. Use v1.rooms instead.");
        return this.v1.rooms;
    }
}
module.exports = Insights;
