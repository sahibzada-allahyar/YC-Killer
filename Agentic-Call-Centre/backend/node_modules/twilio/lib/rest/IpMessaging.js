"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const IpMessagingBase_1 = __importDefault(require("./IpMessagingBase"));
class IpMessaging extends IpMessagingBase_1.default {
    /**
     * @deprecated - Use v2.credentials instead
     */
    get credentials() {
        console.warn("credentials is deprecated. Use v2.credentials instead.");
        return this.v2.credentials;
    }
    /**
     * @deprecated - Use v2.services instead
     */
    get services() {
        console.warn("services is deprecated. Use v2.services instead.");
        return this.v2.services;
    }
}
module.exports = IpMessaging;
