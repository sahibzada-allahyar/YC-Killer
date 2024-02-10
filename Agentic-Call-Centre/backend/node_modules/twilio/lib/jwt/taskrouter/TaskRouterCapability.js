"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Policy = void 0;
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
/**
 * Create a new Policy
 */
class Policy {
    /**
     * Create a new Policy instance
     *
     * @param options - ...
     * @param options.url - Policy URL
     * @param options.method - HTTP Method
     * @param options.queryFilter - Request query filter allowances
     * @param options.postFilter - Request post filter allowances
     * @param options.allowed - Allow the policy
     */
    constructor(options) {
        options = options || {};
        this.url = options.url || "";
        this.method = options.method || "GET";
        this.queryFilter = options.queryFilter || {};
        this.postFilter = options.postFilter || {};
        this.allow = options.allow || true;
    }
    payload() {
        return {
            url: this.url,
            method: this.method,
            query_filter: this.queryFilter,
            post_filter: this.postFilter,
            allow: this.allow,
        };
    }
}
exports.Policy = Policy;
class TaskRouterCapability {
    /**
     * @param options - ...
     * @param options.accountSid - account sid
     * @param options.authToken - auth token
     * @param options.workspaceSid - workspace sid
     * @param options.channelId - taskrouter channel id
     * @param options.friendlyName - friendly name for the jwt
     * @param options.ttl - time to live
     * @param options.version - taskrouter version
     */
    constructor(options) {
        if (!options) {
            throw new Error('Required parameter "options" missing.');
        }
        if (!options.accountSid) {
            throw new Error('Required parameter "options.accountSid" missing.');
        }
        if (!options.authToken) {
            throw new Error('Required parameter "options.authToken" missing.');
        }
        if (!options.workspaceSid) {
            throw new Error('Required parameter "options.workspaceSid" missing.');
        }
        if (!options.channelId) {
            throw new Error('Required parameter "options.channelId" missing.');
        }
        this.accountSid = options.accountSid;
        this.authToken = options.authToken;
        this.workspaceSid = options.workspaceSid;
        this.channelId = options.channelId;
        this.friendlyName = options.friendlyName;
        this.ttl = options.ttl || 3600;
        this.version = options.version || "v1";
        this.policies = [];
    }
    addPolicy(policy) {
        this.policies.push(policy);
    }
    toJwt() {
        var payload = {
            iss: this.accountSid,
            exp: Math.floor(new Date().valueOf() / 1000) + this.ttl,
            version: this.version,
            friendly_name: this.friendlyName,
            account_sid: this.accountSid,
            channel: this.channelId,
            workspace_sid: this.workspaceSid,
            policies: this.policies.map((policy) => policy.payload()),
        };
        if (this.channelId.startsWith("WK")) {
            payload.worker_sid = this.channelId;
        }
        else if (this.channelId.startsWith("WQ")) {
            payload.taskqueue_sid = this.channelId;
        }
        return jsonwebtoken_1.default.sign(payload, this.authToken);
    }
}
exports.default = TaskRouterCapability;
TaskRouterCapability.Policy = Policy;
