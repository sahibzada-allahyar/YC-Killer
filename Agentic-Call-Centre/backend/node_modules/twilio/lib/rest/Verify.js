"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const VerifyBase_1 = __importDefault(require("./VerifyBase"));
class Verify extends VerifyBase_1.default {
    /**
     * @deprecated - Use v2.forms instead
     */
    get forms() {
        console.warn("forms is deprecated. Use v2.forms instead.");
        return this.v2.forms;
    }
    /**
     * @deprecated - Use v2.services instead
     */
    get services() {
        console.warn("services is deprecated. Use v2.services instead.");
        return this.v2.services;
    }
    /**
     * @deprecated - Use v2.verificationAttempts instead
     */
    get verificationAttempts() {
        console.warn("verificationAttempts is deprecated. Use v2.verificationAttempts instead.");
        return this.v2.verificationAttempts;
    }
    /**
     * @deprecated - Use v2.verificationAttemptsSummary instead
     */
    get verificationAttemptsSummary() {
        console.warn("verificationAttemptsSummary is deprecated. Use v2.verificationAttemptsSummary instead.");
        return this.v2.verificationAttemptsSummary;
    }
    /**
     * @deprecated - Use v2.templates instead
     */
    get templates() {
        console.warn("templates is deprecated. Use v2.templates instead.");
        return this.v2.templates;
    }
}
module.exports = Verify;
