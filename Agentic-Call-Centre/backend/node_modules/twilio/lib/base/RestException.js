"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class RestException extends Error {
    constructor(response) {
        super("[HTTP " + response.statusCode + "] Failed to execute request");
        const isResponseBodyString = typeof response.body == "string";
        const body = isResponseBodyString
            ? parseResponseBody(response.body)
            : response.body;
        this.status = response.statusCode;
        if (body !== null) {
            this.message = body.message;
            this.code = body.code;
            this.moreInfo = body.more_info; /* jshint ignore:line */
            this.details = body.details;
        }
        else {
            this.message =
                "[HTTP " + response.statusCode + "] Failed to execute request";
        }
    }
}
exports.default = RestException;
function parseResponseBody(response_body) {
    let body = null;
    try {
        body = JSON.parse(response_body);
    }
    catch (catchError) {
        body = null;
    }
    return body;
}
