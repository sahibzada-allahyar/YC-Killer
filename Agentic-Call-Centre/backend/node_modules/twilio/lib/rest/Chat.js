"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const ChatBase_1 = __importDefault(require("./ChatBase"));
class Chat extends ChatBase_1.default {
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
    /**
     * @deprecated - Use v3.channels instead
     */
    get channels() {
        console.warn("channels is deprecated. Use v3.channels instead.");
        return this.v3.channels;
    }
}
module.exports = Chat;
