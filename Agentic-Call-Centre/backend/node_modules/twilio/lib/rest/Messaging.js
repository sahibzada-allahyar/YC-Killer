"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const MessagingBase_1 = __importDefault(require("./MessagingBase"));
class Messaging extends MessagingBase_1.default {
    /**
     * @deprecated - Use v1.brandRegistrations instead
     */
    get brandRegistrations() {
        console.warn("brandRegistrations is deprecated. Use v1.brandRegistrations instead.");
        return this.v1.brandRegistrations;
    }
    /**
     * @deprecated - Use v1.deactivations instead
     */
    get deactivations() {
        console.warn("deactivations is deprecated. Use v1.deactivations instead.");
        return this.v1.deactivations;
    }
    /**
     * @deprecated - Use v1.domainCerts instead
     */
    get domainCerts() {
        console.warn("domainCerts is deprecated. Use v1.domainCerts instead.");
        return this.v1.domainCerts;
    }
    /**
     * @deprecated - Use v1.domainConfig instead
     */
    get domainConfig() {
        console.warn("domainConfig is deprecated. Use v1.domainConfig instead.");
        return this.v1.domainConfig;
    }
    /**
     * @deprecated - Use v1.externalCampaign instead
     */
    get externalCampaign() {
        console.warn("externalCampaign is deprecated. Use v1.externalCampaign instead.");
        return this.v1.externalCampaign;
    }
    /**
     * @deprecated - Use v1.services instead
     */
    get services() {
        console.warn("services is deprecated. Use v1.services instead.");
        return this.v1.services;
    }
    /**
     * @deprecated - Use v1.usecases instead
     */
    get usecases() {
        console.warn("usecases is deprecated. Use v1.usecases instead.");
        return this.v1.usecases;
    }
}
module.exports = Messaging;
