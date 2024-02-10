"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const SyncBase_1 = __importDefault(require("./SyncBase"));
class Sync extends SyncBase_1.default {
    /**
     * @deprecated - Use v1.services instead
     */
    get services() {
        console.warn("services is deprecated. Use v1.services instead.");
        return this.v1.services;
    }
}
module.exports = Sync;
