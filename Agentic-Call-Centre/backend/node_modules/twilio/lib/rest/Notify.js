"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const NotifyBase_1 = __importDefault(require("./NotifyBase"));
class Notify extends NotifyBase_1.default {
    /**
     * @deprecated - Use v1.credentials instead
     */
    get credentials() {
        console.warn("credentials is deprecated. Use v1.credentials instead.");
        return this.v1.credentials;
    }
    /**
     * @deprecated - Use v1.services instead
     */
    get services() {
        console.warn("services is deprecated. Use v1.services instead.");
        return this.v1.services;
    }
}
module.exports = Notify;
