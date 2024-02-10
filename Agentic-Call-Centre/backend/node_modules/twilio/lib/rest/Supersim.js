"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const SupersimBase_1 = __importDefault(require("./SupersimBase"));
class Supersim extends SupersimBase_1.default {
    /**
     * @deprecated - Use v1.esimProfiles instead
     */
    get esimProfiles() {
        console.warn("esimProfiles is deprecated. Use v1.esimProfiles instead.");
        return this.v1.esimProfiles;
    }
    /**
     * @deprecated - Use v1.fleets instead
     */
    get fleets() {
        console.warn("fleets is deprecated. Use v1.fleets instead.");
        return this.v1.fleets;
    }
    /**
     * @deprecated - Use v1.ipCommands instead
     */
    get ipCommands() {
        console.warn("ipCommands is deprecated. Use v1.ipCommands instead.");
        return this.v1.ipCommands;
    }
    /**
     * @deprecated - Use v1.networks instead
     */
    get networks() {
        console.warn("networks is deprecated. Use v1.networks instead.");
        return this.v1.networks;
    }
    /**
     * @deprecated - Use v1.settingsUpdates instead
     */
    get settingsUpdates() {
        console.warn("settingsUpdates is deprecated. Use v1.settingsUpdates instead.");
        return this.v1.settingsUpdates;
    }
    /**
     * @deprecated - Use v1.networkAccessProfiles instead
     */
    get networkAccessProfiles() {
        console.warn("networkAccessProfiles is deprecated. Use v1.networkAccessProfiles instead.");
        return this.v1.networkAccessProfiles;
    }
    /**
     * @deprecated - Use v1.sims instead
     */
    get sims() {
        console.warn("sims is deprecated. Use v1.sims instead.");
        return this.v1.sims;
    }
    /**
     * @deprecated - Use v1.smsCommands instead
     */
    get smsCommands() {
        console.warn("smsCommands is deprecated. Use v1.smsCommands instead.");
        return this.v1.smsCommands;
    }
    /**
     * @deprecated - Use v1.usageRecords instead
     */
    get usageRecords() {
        console.warn("usageRecords is deprecated. Use v1.usageRecords instead.");
        return this.v1.usageRecords;
    }
}
module.exports = Supersim;
