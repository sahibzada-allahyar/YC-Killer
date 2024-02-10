"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const ContentBase_1 = __importDefault(require("./ContentBase"));
class Content extends ContentBase_1.default {
    /**
     * @deprecated - Use v1.contents instead
     */
    get contents() {
        console.warn("contents is deprecated. Use v1.contents instead.");
        return this.v1.contents;
    }
}
module.exports = Content;
