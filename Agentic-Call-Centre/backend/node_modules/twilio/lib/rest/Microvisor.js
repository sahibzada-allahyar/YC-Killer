"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const MicrovisorBase_1 = __importDefault(require("./MicrovisorBase"));
class Microvisor extends MicrovisorBase_1.default {
    /**
     * @deprecated - Use v1.apps instead
     */
    get apps() {
        console.warn("apps is deprecated. Use v1.apps instead.");
        return this.v1.apps;
    }
    /**
     * @deprecated - Use v1.devices instead
     */
    get devices() {
        console.warn("devices is deprecated. Use v1.devices instead.");
        return this.v1.devices;
    }
}
module.exports = Microvisor;
