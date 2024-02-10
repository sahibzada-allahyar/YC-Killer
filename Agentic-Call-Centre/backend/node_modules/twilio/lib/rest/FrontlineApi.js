"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const FrontlineApiBase_1 = __importDefault(require("./FrontlineApiBase"));
class FrontlineApi extends FrontlineApiBase_1.default {
    /**
     * @deprecated - Use v1.users instead
     */
    get users() {
        console.warn("users is deprecated. Use v1.users instead.");
        return this.v1.users;
    }
}
module.exports = FrontlineApi;
