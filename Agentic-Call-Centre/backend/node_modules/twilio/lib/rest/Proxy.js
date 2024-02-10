"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const ProxyBase_1 = __importDefault(require("./ProxyBase"));
class Proxy extends ProxyBase_1.default {
    /**
     * @deprecated - Use v1.services instead
     */
    get services() {
        console.warn("services is deprecated. Use v1.services instead.");
        return this.v1.services;
    }
}
module.exports = Proxy;
