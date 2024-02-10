"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const StudioBase_1 = __importDefault(require("./StudioBase"));
class Studio extends StudioBase_1.default {
    /**
     * @deprecated - Use v2.flows instead
     */
    get flows() {
        console.warn("flows is deprecated. Use v2.flows instead.");
        return this.v2.flows;
    }
    /**
     * @deprecated - Use v2.flowValidate instead
     */
    get flowValidate() {
        console.warn("flowValidate is deprecated. Use v2.flowValidate instead.");
        return this.v2.flowValidate;
    }
}
module.exports = Studio;
