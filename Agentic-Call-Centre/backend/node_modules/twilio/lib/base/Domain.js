"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const utility_1 = require("./utility");
/**
 * Base domain object
 */
class Domain {
    /**
     * Creates a Domain instance
     *
     * @param twilio - A Twilio Client
     * @param baseUrl - Base url for this domain
     */
    constructor(twilio, baseUrl) {
        this.twilio = twilio;
        this.baseUrl = baseUrl;
    }
    /**
     * Turn a uri into an absolute url
     *
     * @param uri - uri to transform
     * @returns absolute url
     */
    absoluteUrl(uri) {
        var result = "";
        if (typeof this.baseUrl === "string") {
            const cleanBaseUrl = (0, utility_1.trim)(this.baseUrl, "/");
            result += cleanBaseUrl;
            result += "/";
        }
        if (typeof uri === "string") {
            uri = (0, utility_1.trim)(uri, "/");
            if (result === "") {
                result += "/";
            }
            result += uri;
        }
        return result;
    }
    /**
     * Make request to this domain
     *
     * @param opts - request options
     * @returns request promise
     */
    request(opts) {
        return this.twilio.request({
            ...opts,
            uri: this.absoluteUrl(opts.uri),
        });
    }
}
exports.default = Domain;
