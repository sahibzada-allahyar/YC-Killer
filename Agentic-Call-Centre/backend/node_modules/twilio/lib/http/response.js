"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class Response {
    constructor(statusCode, body, headers) {
        this.statusCode = statusCode;
        this.body = body;
        this.headers = headers;
    }
    toString() {
        return "HTTP " + this.statusCode + " " + this.body;
    }
}
exports.default = Response;
