"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const PreviewBase_1 = __importDefault(require("./PreviewBase"));
class Preview extends PreviewBase_1.default {
    /**
     * @deprecated - Use deployed_devices.fleets instead
     */
    get fleets() {
        console.warn("fleets is deprecated. Use deployed_devices.fleets instead.");
        return this.deployed_devices.fleets;
    }
    /**
     * @deprecated - Use hosted_numbers.authorizationDocuments instead
     */
    get authorizationDocuments() {
        console.warn("authorizationDocuments is deprecated. Use hosted_numbers.authorizationDocuments instead.");
        return this.hosted_numbers.authorizationDocuments;
    }
    /**
     * @deprecated - Use hosted_numbers.hostedNumberOrders instead
     */
    get hostedNumberOrders() {
        console.warn("hostedNumberOrders is deprecated. Use hosted_numbers.hostedNumberOrders instead.");
        return this.hosted_numbers.hostedNumberOrders;
    }
    /**
     * @deprecated - Use marketplace.availableAddOns instead
     */
    get availableAddOns() {
        console.warn("availableAddOns is deprecated. Use marketplace.availableAddOns instead.");
        return this.marketplace.availableAddOns;
    }
    /**
     * @deprecated - Use marketplace.installedAddOns instead
     */
    get installedAddOns() {
        console.warn("installedAddOns is deprecated. Use marketplace.installedAddOns instead.");
        return this.marketplace.installedAddOns;
    }
    /**
     * @deprecated - Use sync.services instead
     */
    get services() {
        console.warn("services is deprecated. Use sync.services instead.");
        return this.sync.services;
    }
    /**
     * @deprecated - Use wireless.commands instead
     */
    get commands() {
        console.warn("commands is deprecated. Use wireless.commands instead.");
        return this.wireless.commands;
    }
    /**
     * @deprecated - Use wireless.ratePlans instead
     */
    get ratePlans() {
        console.warn("ratePlans is deprecated. Use wireless.ratePlans instead.");
        return this.wireless.ratePlans;
    }
    /**
     * @deprecated - Use wireless.sims instead
     */
    get sims() {
        console.warn("sims is deprecated. Use wireless.sims instead.");
        return this.wireless.sims;
    }
}
module.exports = Preview;
