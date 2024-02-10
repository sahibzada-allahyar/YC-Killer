"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const AccountsBase_1 = __importDefault(require("./AccountsBase"));
class Accounts extends AccountsBase_1.default {
    /**
     * @deprecated - Use v1.authTokenPromotion; instead
     */
    get authTokenPromotion() {
        console.warn("authTokenPromotion is deprecated. Use v1.authTokenPromotion; instead.");
        return this.v1.authTokenPromotion;
    }
    /**
     * @deprecated - Use v1.credentials; instead
     */
    get credentials() {
        console.warn("credentials is deprecated. Use v1.credentials; instead.");
        return this.v1.credentials;
    }
    /**
     * @deprecated - Use v1.secondaryAuthToken; instead
     */
    get secondaryAuthToken() {
        console.warn("secondaryAuthToken is deprecated. Use v1.secondaryAuthToken; instead.");
        return this.v1.secondaryAuthToken;
    }
}
module.exports = Accounts;
