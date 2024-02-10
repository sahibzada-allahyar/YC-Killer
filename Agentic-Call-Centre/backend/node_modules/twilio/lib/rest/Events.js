"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const EventsBase_1 = __importDefault(require("./EventsBase"));
class Events extends EventsBase_1.default {
    /**
     * @deprecated - Use v1.eventTypes instead
     */
    get eventTypes() {
        console.warn("eventTypes is deprecated. Use v1.eventTypes instead.");
        return this.v1.eventTypes;
    }
    /**
     * @deprecated - Use v1.schemas instead
     */
    get schemas() {
        console.warn("schemas is deprecated. Use v1.schemas instead.");
        return this.v1.schemas;
    }
    /**
     * @deprecated - Use v1.sinks instead
     */
    get sinks() {
        console.warn("sinks is deprecated. Use v1.sinks instead.");
        return this.v1.sinks;
    }
    /**
     * @deprecated - Use v1.subscriptions instead
     */
    get subscriptions() {
        console.warn("subscriptions is deprecated. Use v1.subscriptions instead.");
        return this.v1.subscriptions;
    }
}
module.exports = Events;
