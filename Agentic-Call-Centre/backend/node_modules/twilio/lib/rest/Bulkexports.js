"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const BulkexportsBase_1 = __importDefault(require("./BulkexportsBase"));
class Bulkexports extends BulkexportsBase_1.default {
    /**
     * @deprecated - Use v1.exports instead
     */
    get exports() {
        console.warn("exports is deprecated. Use v1.exports instead.");
        return this.v1.exports;
    }
    /**
     * @deprecated - Use v1.exportConfiguration instead
     */
    get exportConfiguration() {
        console.warn("exportConfiguration is deprecated. Use v1.exportConfiguration instead.");
        return this.v1.exportConfiguration;
    }
}
module.exports = Bulkexports;
