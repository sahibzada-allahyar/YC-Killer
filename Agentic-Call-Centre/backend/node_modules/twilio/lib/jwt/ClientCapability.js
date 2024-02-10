"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.OutgoingClientScope = exports.IncomingClientScope = exports.EventStreamScope = void 0;
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const querystring_1 = __importDefault(require("querystring"));
/**
 * @param filters
 */
class EventStreamScope {
    constructor(filters) {
        this.scope = "scope:stream:subscribe";
        this.filters = filters || {};
    }
    payload() {
        var queryArgs = ["path=/2010-04-01/Events"];
        if (Object.keys(this.filters).length > 0) {
            var queryParams = Object.entries(this.filters).map((filter) => {
                return [querystring_1.default.escape(filter[0]), querystring_1.default.escape(filter[1])].join("=");
            });
            var filterParams = queryParams.join("&");
            queryArgs.push(["appParams", querystring_1.default.escape(filterParams)].join("="));
        }
        var queryString = queryArgs.join("&");
        return [this.scope, queryString].join("?");
    }
}
exports.EventStreamScope = EventStreamScope;
/**
 * @param clientName
 */
class IncomingClientScope {
    constructor(clientName) {
        this.scope = "scope:client:incoming";
        this.clientName = clientName;
    }
    payload() {
        var query = ["clientName", querystring_1.default.escape(this.clientName)].join("=");
        return [this.scope, query].join("?");
    }
}
exports.IncomingClientScope = IncomingClientScope;
class OutgoingClientScope {
    /**
     * @param options - ...
     * @param options.applicationSid - the application sid
     * @param options.clientName - the client name
     * @param options.params - parameters
     */
    constructor(options) {
        this.scope = "scope:client:outgoing";
        if (!options) {
            throw new Error('Required parameter "options" missing.');
        }
        if (typeof options !== "object") {
            throw new TypeError('Parameter "options" must be a type Object');
        }
        if (!options.applicationSid) {
            throw new Error('Required parameter "options.applicationSid" missing.');
        }
        this.applicationSid = options.applicationSid;
        this.clientName = options.clientName;
        this.params = options.params;
    }
    payload() {
        var queryArgs = [["appSid", querystring_1.default.escape(this.applicationSid)].join("=")];
        if (typeof this.clientName === "string") {
            queryArgs.push(["clientName", querystring_1.default.escape(this.clientName)].join("="));
        }
        if (typeof this.params === "object") {
            var queryParams = Object.entries(this.params).map((param) => {
                return [querystring_1.default.escape(param[0]), querystring_1.default.escape(param[1])].join("=");
            });
            var filterParams = queryParams.join("&");
            queryArgs.push(["appParams", querystring_1.default.escape(filterParams)].join("="));
        }
        var queryString = queryArgs.join("&");
        return [this.scope, queryString].join("?");
    }
}
exports.OutgoingClientScope = OutgoingClientScope;
/**
 * @param options
 */
class ClientCapability {
    constructor(options) {
        if (!options) {
            throw new Error('Required parameter "options" missing.');
        }
        if (typeof options !== "object") {
            throw new TypeError('Parameter "options" must be a type Object');
        }
        if (!options.accountSid) {
            throw new Error('Required parameter "options.accountSid" missing.');
        }
        if (!options.authToken) {
            throw new Error('Required parameter "options.authToken" missing.');
        }
        this.accountSid = options.accountSid;
        this.authToken = options.authToken;
        this.ttl = options.ttl || 3600;
        this.scopes = [];
    }
    addScope(scope) {
        this.scopes.push(scope);
    }
    toJwt() {
        const scope = this.scopes.map((scope) => scope.payload()).join(" ");
        var payload = {
            scope: scope,
            iss: this.accountSid,
            exp: Math.floor(new Date().valueOf() / 1000) + this.ttl,
        };
        return jsonwebtoken_1.default.sign(payload, this.authToken);
    }
}
exports.default = ClientCapability;
ClientCapability.EventStreamScope = EventStreamScope;
ClientCapability.IncomingClientScope = IncomingClientScope;
ClientCapability.OutgoingClientScope = OutgoingClientScope;
