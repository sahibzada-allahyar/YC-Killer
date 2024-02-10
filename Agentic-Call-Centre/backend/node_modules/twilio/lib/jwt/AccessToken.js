"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
class AccessToken {
    /**
     * @param accountSid - The account's unique ID to which access is scoped
     * @param keySid - The signing key's unique ID
     * @param secret - The secret to sign the token with
     * @param options - ...
     * @param options.ttl - Time to live in seconds (default 3600)
     * @param options.identity - The identity of the first person. Required.
     * @param options.nbf - Time from epoch in seconds for not before value
     * @param options.region - The region value associated with this account
     */
    constructor(accountSid, keySid, secret, options) {
        if (!accountSid) {
            throw new Error("accountSid is required");
        }
        if (!keySid) {
            throw new Error("keySid is required");
        }
        if (!secret) {
            throw new Error("secret is required");
        }
        if (!options || !options.identity) {
            throw new Error("identity is required to be specified in options");
        }
        this.accountSid = accountSid;
        this.keySid = keySid;
        this.secret = secret;
        this.ttl = options.ttl || 3600;
        this.identity = options.identity;
        this.nbf = options.nbf;
        this.region = options.region;
        this.grants = [];
    }
    addGrant(grant) {
        this.grants.push(grant);
    }
    toJwt(algorithm) {
        algorithm = algorithm || AccessToken.DEFAULT_ALGORITHM;
        if (!AccessToken.ALGORITHMS.includes(algorithm)) {
            throw new Error("Algorithm not supported. Allowed values are " +
                AccessToken.ALGORITHMS.join(", "));
        }
        let grants = {};
        if (Number.isInteger(this.identity) || typeof this.identity === "string") {
            grants.identity = String(this.identity);
        }
        for (const grant of this.grants) {
            grants[grant.key] = grant.toPayload();
        }
        const now = Math.floor(Date.now() / 1000);
        let payload = {
            jti: this.keySid + "-" + now,
            grants: grants,
        };
        if (typeof this.nbf === "number") {
            payload.nbf = this.nbf;
        }
        let header = {
            cty: "twilio-fpa;v=1",
            typ: "JWT",
        };
        if (this.region && typeof this.region === "string") {
            header.twr = this.region;
        }
        return jsonwebtoken_1.default.sign(payload, this.secret, {
            header: header,
            algorithm: algorithm,
            issuer: this.keySid,
            subject: this.accountSid,
            expiresIn: this.ttl,
        });
    }
}
AccessToken.DEFAULT_ALGORITHM = "HS256";
AccessToken.ALGORITHMS = ["HS256", "HS384", "HS512"];
(function (AccessToken) {
    class Grant {
        constructor(key) {
            this.key = key;
        }
    }
    AccessToken.Grant = Grant;
    class TaskRouterGrant extends Grant {
        /**
         * @param options - ...
         * @param options.workspaceSid - The workspace unique ID
         * @param options.workerSid - The worker unique ID
         * @param options.role - The role of the grant
         */
        constructor(options) {
            options = options || {};
            super("task_router");
            this.workspaceSid = options.workspaceSid;
            this.workerSid = options.workerSid;
            this.role = options.role;
        }
        toPayload() {
            let grant = {};
            if (this.workspaceSid) {
                grant.workspace_sid = this.workspaceSid;
            }
            if (this.workerSid) {
                grant.worker_sid = this.workerSid;
            }
            if (this.role) {
                grant.role = this.role;
            }
            return grant;
        }
    }
    AccessToken.TaskRouterGrant = TaskRouterGrant;
    class ChatGrant extends Grant {
        /**
         * @param options - ...
         * @param options.serviceSid - The service unique ID
         * @param options.endpointId - The endpoint ID
         * @param options.deploymentRoleSid - SID of the deployment role to be
         *                 assigned to the user
         * @param options.pushCredentialSid - The Push Credentials SID
         */
        constructor(options) {
            options = options || {};
            super("chat");
            this.serviceSid = options.serviceSid;
            this.endpointId = options.endpointId;
            this.deploymentRoleSid = options.deploymentRoleSid;
            this.pushCredentialSid = options.pushCredentialSid;
        }
        toPayload() {
            let grant = {};
            if (this.serviceSid) {
                grant.service_sid = this.serviceSid;
            }
            if (this.endpointId) {
                grant.endpoint_id = this.endpointId;
            }
            if (this.deploymentRoleSid) {
                grant.deployment_role_sid = this.deploymentRoleSid;
            }
            if (this.pushCredentialSid) {
                grant.push_credential_sid = this.pushCredentialSid;
            }
            return grant;
        }
    }
    AccessToken.ChatGrant = ChatGrant;
    class VideoGrant extends Grant {
        /**
         * @param options - ...
         * @param options.room - The Room name or Room sid.
         */
        constructor(options) {
            options = options || {};
            super("video");
            this.room = options.room;
        }
        toPayload() {
            let grant = {};
            if (this.room) {
                grant.room = this.room;
            }
            return grant;
        }
    }
    AccessToken.VideoGrant = VideoGrant;
    class SyncGrant extends Grant {
        /**
         * @param options.serviceSid - The service unique ID
         * @param options.endpointId - The endpoint ID
         */
        constructor(options) {
            options = options || {};
            super("data_sync");
            this.serviceSid = options.serviceSid;
            this.endpointId = options.endpointId;
        }
        toPayload() {
            let grant = {};
            if (this.serviceSid) {
                grant.service_sid = this.serviceSid;
            }
            if (this.endpointId) {
                grant.endpoint_id = this.endpointId;
            }
            return grant;
        }
    }
    AccessToken.SyncGrant = SyncGrant;
    class VoiceGrant extends Grant {
        /**
         * @param options - ...
         * @param options.incomingAllow - Whether or not this endpoint is allowed to receive incoming calls as grants.identity
         * @param options.outgoingApplicationSid - application sid to call when placing outgoing call
         * @param options.outgoingApplicationParams - request params to pass to the application
         * @param options.pushCredentialSid - Push Credential Sid to use when registering to receive incoming call notifications
         * @param options.endpointId - Specify an endpoint identifier for this device, which will allow the developer
         *                 to direct calls to a specific endpoint when multiple devices are associated with a single identity
         */
        constructor(options) {
            options = options || {};
            super("voice");
            this.incomingAllow = options.incomingAllow;
            this.outgoingApplicationSid = options.outgoingApplicationSid;
            this.outgoingApplicationParams = options.outgoingApplicationParams;
            this.pushCredentialSid = options.pushCredentialSid;
            this.endpointId = options.endpointId;
        }
        toPayload() {
            let grant = {};
            if (this.incomingAllow === true) {
                grant.incoming = { allow: true };
            }
            if (this.outgoingApplicationSid) {
                grant.outgoing = {
                    application_sid: this.outgoingApplicationSid,
                };
                if (this.outgoingApplicationParams) {
                    grant.outgoing.params = this.outgoingApplicationParams;
                }
            }
            if (this.pushCredentialSid) {
                grant.push_credential_sid = this.pushCredentialSid;
            }
            if (this.endpointId) {
                grant.endpoint_id = this.endpointId;
            }
            return grant;
        }
    }
    AccessToken.VoiceGrant = VoiceGrant;
    class PlaybackGrant extends Grant {
        /**
         * @param options - ...
         * @param options.grant - The PlaybackGrant retrieved from Twilio's API
         */
        constructor(options) {
            options = options || {};
            super("player");
            this.grant = options.grant;
        }
        toPayload() {
            let grant = {};
            if (this.grant) {
                grant = this.grant;
            }
            return grant;
        }
    }
    AccessToken.PlaybackGrant = PlaybackGrant;
})(AccessToken || (AccessToken = {}));
module.exports = AccessToken;
