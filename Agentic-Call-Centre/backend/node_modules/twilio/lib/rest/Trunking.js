"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const TrunkingBase_1 = __importDefault(require("./TrunkingBase"));
class Trunking extends TrunkingBase_1.default {
    /**
     * @deprecated - Use v1.trunks instead
     */
    get trunks() {
        console.warn("trunks is deprecated. Use v1.trunks instead.");
        return this.v1.trunks;
    }
}
module.exports = Trunking;
