"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const MonitorBase_1 = __importDefault(require("./MonitorBase"));
class Monitor extends MonitorBase_1.default {
    /**
     * @deprecated - Use v1.alerts instead
     */
    get alerts() {
        console.warn("alerts is deprecated. Use v1.alerts instead.");
        return this.v1.alerts;
    }
    /**
     * @deprecated - Use v1.events instead
     */
    get events() {
        console.warn("events is deprecated. Use v1.events instead.");
        return this.v1.events;
    }
}
module.exports = Monitor;
