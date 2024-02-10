"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const VoiceBase_1 = __importDefault(require("./VoiceBase"));
class Voice extends VoiceBase_1.default {
    /**
     * @deprecated - Use v1.archivedCalls instead
     */
    get archivedCalls() {
        console.warn("archivedCalls is deprecated. Use v1.archivedCalls instead.");
        return this.v1.archivedCalls;
    }
    /**
     * @deprecated - Use v1.byocTrunks instead
     */
    get byocTrunks() {
        console.warn("byocTrunks is deprecated. Use v1.byocTrunks instead.");
        return this.v1.byocTrunks;
    }
    /**
     * @deprecated - Use v1.connectionPolicies instead
     */
    get connectionPolicies() {
        console.warn("connectionPolicies is deprecated. Use v1.connectionPolicies instead.");
        return this.v1.connectionPolicies;
    }
    /**
     * @deprecated - Use v1.dialingPermissions instead
     */
    get dialingPermissions() {
        console.warn("dialingPermissions is deprecated. Use v1.dialingPermissions instead.");
        return this.v1.dialingPermissions;
    }
    /**
     * @deprecated - Use v1.ipRecords instead
     */
    get ipRecords() {
        console.warn("ipRecords is deprecated. Use v1.ipRecords instead.");
        return this.v1.ipRecords;
    }
    /**
     * @deprecated - Use v1.sourceIpMappings instead
     */
    get sourceIpMappings() {
        console.warn("sourceIpMappings is deprecated. Use v1.sourceIpMappings instead.");
        return this.v1.sourceIpMappings;
    }
}
module.exports = Voice;
