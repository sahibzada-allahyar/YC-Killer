"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const TrusthubBase_1 = __importDefault(require("./TrusthubBase"));
class Trusthub extends TrusthubBase_1.default {
    /**
     * @deprecated - Use v1.customerProfiles instead
     */
    get customerProfiles() {
        console.warn("customerProfiles is deprecated. Use v1.customerProfiles instead.");
        return this.v1.customerProfiles;
    }
    /**
     * @deprecated - Use v1.endUsers instead
     */
    get endUsers() {
        console.warn("endUsers is deprecated. Use v1.endUsers instead.");
        return this.v1.endUsers;
    }
    /**
     * @deprecated - Use v1.endUserTypes instead
     */
    get endUserTypes() {
        console.warn("endUserTypes is deprecated. Use v1.endUserTypes instead.");
        return this.v1.endUserTypes;
    }
    /**
     * @deprecated - Use v1.policies instead
     */
    get policies() {
        console.warn("policies is deprecated. Use v1.policies instead.");
        return this.v1.policies;
    }
    /**
     * @deprecated - Use v1.supportingDocuments instead
     */
    get supportingDocuments() {
        console.warn("supportingDocuments is deprecated. Use v1.supportingDocuments instead.");
        return this.v1.supportingDocuments;
    }
    /**
     * @deprecated - Use v1.supportingDocumentTypes instead
     */
    get supportingDocumentTypes() {
        console.warn("supportingDocumentTypes is deprecated. Use v1.supportingDocumentTypes instead.");
        return this.v1.supportingDocumentTypes;
    }
    /**
     * @deprecated - Use v1.trustProducts instead
     */
    get trustProducts() {
        console.warn("trustProducts is deprecated. Use v1.trustProducts instead.");
        return this.v1.trustProducts;
    }
}
module.exports = Trusthub;
