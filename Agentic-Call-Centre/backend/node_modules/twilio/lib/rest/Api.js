"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const ApiBase_1 = __importDefault(require("./ApiBase"));
class Api extends ApiBase_1.default {
    get account() {
        return this.v2010.account;
    }
    get accounts() {
        return this.v2010.accounts;
    }
    /**
     * @deprecated - Use account.addresses instead
     */
    get addresses() {
        console.warn("addresses is deprecated. Use account.addresses instead.");
        return this.account.addresses;
    }
    /**
     * @deprecated - Use account.applications instead
     */
    get applications() {
        console.warn("applications is deprecated. Use account.applications instead.");
        return this.account.applications;
    }
    /**
     * @deprecated - Use account.authorizedConnectApps instead
     */
    get authorizedConnectApps() {
        console.warn("authorizedConnectApps is deprecated. Use account.authorizedConnectApps instead.");
        return this.account.authorizedConnectApps;
    }
    /**
     * @deprecated - Use account.availablePhoneNumbers instead
     */
    get availablePhoneNumbers() {
        console.warn("availablePhoneNumbers is deprecated. Use account.availablePhoneNumbers instead.");
        return this.account.availablePhoneNumbers;
    }
    /**
     * @deprecated - Use account.balance instead
     */
    get balance() {
        console.warn("balance is deprecated. Use account.balance instead.");
        return this.account.balance;
    }
    /**
     * @deprecated - Use account.calls instead
     */
    get calls() {
        console.warn("calls is deprecated. Use account.calls instead.");
        return this.account.calls;
    }
    /**
     * @deprecated - Use account.conferences instead
     */
    get conferences() {
        console.warn("conferences is deprecated. Use account.conferences instead.");
        return this.account.conferences;
    }
    /**
     * @deprecated - Use account.connectApps instead
     */
    get connectApps() {
        console.warn("connectApps is deprecated. Use account.connectApps instead.");
        return this.account.connectApps;
    }
    /**
     * @deprecated - Use account.incomingPhoneNumbers instead
     */
    get incomingPhoneNumbers() {
        console.warn("incomingPhoneNumbers is deprecated. Use account.incomingPhoneNumbers instead.");
        return this.account.incomingPhoneNumbers;
    }
    /**
     * @deprecated - Use account.keys instead
     */
    get keys() {
        console.warn("keys is deprecated. Use account.keys instead.");
        return this.account.keys;
    }
    /**
     * @deprecated - Use account.messages instead
     */
    get messages() {
        console.warn("messages is deprecated. Use account.messages instead.");
        return this.account.messages;
    }
    /**
     * @deprecated - Use account.newKeys instead
     */
    get newKeys() {
        console.warn("newKeys is deprecated. Use account.newKeys instead.");
        return this.account.newKeys;
    }
    /**
     * @deprecated - Use account.newSigningKeys instead
     */
    get newSigningKeys() {
        console.warn("newSigningKeys is deprecated. Use account.newSigningKeys instead.");
        return this.account.newSigningKeys;
    }
    /**
     * @deprecated - Use account.notifications instead
     */
    get notifications() {
        console.warn("notifications is deprecated. Use account.notifications instead.");
        return this.account.notifications;
    }
    /**
     * @deprecated - Use account.outgoingCallerIds instead
     */
    get outgoingCallerIds() {
        console.warn("outgoingCallerIds is deprecated. Use account.outgoingCallerIds instead.");
        return this.account.outgoingCallerIds;
    }
    /**
     * @deprecated - Use account.queues instead
     */
    get queues() {
        console.warn("queues is deprecated. Use account.queues instead.");
        return this.account.queues;
    }
    /**
     * @deprecated - Use account.recordings instead
     */
    get recordings() {
        console.warn("recordings is deprecated. Use account.recordings instead.");
        return this.account.recordings;
    }
    /**
     * @deprecated - Use account.signingKeys instead
     */
    get signingKeys() {
        console.warn("signingKeys is deprecated. Use account.signingKeys instead.");
        return this.account.signingKeys;
    }
    /**
     * @deprecated - Use account.sip instead
     */
    get sip() {
        console.warn("sip is deprecated. Use account.sip instead.");
        return this.account.sip;
    }
    /**
     * @deprecated - Use account.shortCodes instead
     */
    get shortCodes() {
        console.warn("shortCodes is deprecated. Use account.shortCodes instead.");
        return this.account.shortCodes;
    }
    /**
     * @deprecated - Use account.tokens instead
     */
    get tokens() {
        console.warn("tokens is deprecated. Use account.tokens instead.");
        return this.account.tokens;
    }
    /**
     * @deprecated - Use account.transcriptions instead
     */
    get transcriptions() {
        console.warn("transcriptions is deprecated. Use account.transcriptions instead.");
        return this.account.transcriptions;
    }
    /**
     * @deprecated - Use account.usage instead
     */
    get usage() {
        console.warn("usage is deprecated. Use account.usage instead.");
        return this.account.usage;
    }
    /**
     * @deprecated - Use account.validationRequests instead
     */
    get validationRequests() {
        console.warn("validationRequests is deprecated. Use account.validationRequests instead.");
        return this.account.validationRequests;
    }
}
module.exports = Api;
