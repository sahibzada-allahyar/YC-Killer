"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const axios_1 = __importDefault(require("axios"));
const fs = __importStar(require("fs"));
const https_proxy_agent_1 = __importDefault(require("https-proxy-agent"));
const qs_1 = __importDefault(require("qs"));
const https = __importStar(require("https"));
const response_1 = __importDefault(require("../http/response"));
const request_1 = __importDefault(require("../http/request"));
const DEFAULT_CONTENT_TYPE = "application/x-www-form-urlencoded";
const DEFAULT_TIMEOUT = 30000;
const DEFAULT_INITIAL_RETRY_INTERVAL_MILLIS = 100;
const DEFAULT_MAX_RETRY_DELAY = 3000;
const DEFAULT_MAX_RETRIES = 3;
function getExponentialBackoffResponseHandler(axios, opts) {
    const maxIntervalMillis = opts.maxIntervalMillis;
    const maxRetries = opts.maxRetries;
    return function (res) {
        const config = res.config;
        if (res.status !== 429) {
            return res;
        }
        const retryCount = (config.retryCount || 0) + 1;
        if (retryCount <= maxRetries) {
            config.retryCount = retryCount;
            const baseDelay = Math.min(maxIntervalMillis, DEFAULT_INITIAL_RETRY_INTERVAL_MILLIS * Math.pow(2, retryCount));
            const delay = Math.floor(baseDelay * Math.random()); // Full jitter backoff
            return new Promise((resolve) => {
                setTimeout(() => resolve(axios(config)), delay);
            });
        }
        return res;
    };
}
class RequestClient {
    /**
     * Make http request
     * @param opts - The options passed to https.Agent
     * @param opts.timeout - https.Agent timeout option. Used as the socket timeout, AND as the default request timeout.
     * @param opts.keepAlive - https.Agent keepAlive option
     * @param opts.keepAliveMsecs - https.Agent keepAliveMsecs option
     * @param opts.maxSockets - https.Agent maxSockets option
     * @param opts.maxTotalSockets - https.Agent maxTotalSockets option
     * @param opts.maxFreeSockets - https.Agent maxFreeSockets option
     * @param opts.scheduling - https.Agent scheduling option
     * @param opts.autoRetry - Enable auto-retry requests with exponential backoff on 429 responses. Defaults to false.
     * @param opts.maxRetryDelay - Max retry delay in milliseconds for 429 Too Many Request response retries. Defaults to 3000.
     * @param opts.maxRetries - Max number of request retries for 429 Too Many Request responses. Defaults to 3.
     */
    constructor(opts) {
        opts = opts || {};
        this.defaultTimeout = opts.timeout || DEFAULT_TIMEOUT;
        this.autoRetry = opts.autoRetry || false;
        this.maxRetryDelay = opts.maxRetryDelay || DEFAULT_MAX_RETRY_DELAY;
        this.maxRetries = opts.maxRetries || DEFAULT_MAX_RETRIES;
        // construct an https agent
        let agentOpts = {
            timeout: this.defaultTimeout,
            keepAlive: opts.keepAlive,
            keepAliveMsecs: opts.keepAliveMsecs,
            maxSockets: opts.maxSockets,
            maxTotalSockets: opts.maxTotalSockets,
            maxFreeSockets: opts.maxFreeSockets,
            scheduling: opts.scheduling,
            ca: opts.ca,
        };
        // sets https agent CA bundle if defined in CA bundle filepath env variable
        if (process.env.TWILIO_CA_BUNDLE !== undefined) {
            if (agentOpts.ca === undefined) {
                agentOpts.ca = fs.readFileSync(process.env.TWILIO_CA_BUNDLE);
            }
        }
        let agent;
        if (process.env.HTTP_PROXY) {
            // Note: if process.env.HTTP_PROXY is set, we're not able to apply the given
            // socket timeout. See: https://github.com/TooTallNate/node-https-proxy-agent/pull/96
            agent = (0, https_proxy_agent_1.default)(process.env.HTTP_PROXY);
        }
        else {
            agent = new https.Agent(agentOpts);
        }
        // construct an axios instance
        this.axios = axios_1.default.create();
        this.axios.defaults.headers.post["Content-Type"] = DEFAULT_CONTENT_TYPE;
        this.axios.defaults.httpsAgent = agent;
        if (opts.autoRetry) {
            this.axios.interceptors.response.use(getExponentialBackoffResponseHandler(this.axios, {
                maxIntervalMillis: this.maxRetryDelay,
                maxRetries: this.maxRetries,
            }));
        }
    }
    /**
     * Make http request
     * @param opts - The options argument
     * @param opts.method - The http method
     * @param opts.uri - The request uri
     * @param opts.username - The username used for auth
     * @param opts.password - The password used for auth
     * @param opts.headers - The request headers
     * @param opts.params - The request params
     * @param opts.data - The request data
     * @param opts.timeout - The request timeout in milliseconds (default 30000)
     * @param opts.allowRedirects - Should the client follow redirects
     * @param opts.forever - Set to true to use the forever-agent
     * @param opts.logLevel - Show debug logs
     */
    request(opts) {
        if (!opts.method) {
            throw new Error("http method is required");
        }
        if (!opts.uri) {
            throw new Error("uri is required");
        }
        var headers = opts.headers || {};
        if (!headers.Connection && !headers.connection && opts.forever) {
            headers.Connection = "keep-alive";
        }
        else if (!headers.Connection && !headers.connection) {
            headers.Connection = "close";
        }
        let auth = undefined;
        if (opts.username && opts.password) {
            auth = Buffer.from(opts.username + ":" + opts.password).toString("base64");
            headers.Authorization = "Basic " + auth;
        }
        const options = {
            timeout: opts.timeout || this.defaultTimeout,
            maxRedirects: opts.allowRedirects ? 10 : 0,
            url: opts.uri,
            method: opts.method,
            headers: opts.headers,
            proxy: false,
            validateStatus: (status) => status >= 100 && status < 600,
        };
        if (opts.data) {
            options.data = qs_1.default.stringify(opts.data, { arrayFormat: "repeat" });
        }
        if (opts.params) {
            options.params = opts.params;
            options.paramsSerializer = (params) => {
                return qs_1.default.stringify(params, { arrayFormat: "repeat" });
            };
        }
        const requestOptions = {
            method: opts.method,
            url: opts.uri,
            auth: auth,
            params: options.params,
            data: opts.data,
            headers: opts.headers,
        };
        if (opts.logLevel === "debug") {
            this.logRequest(requestOptions);
        }
        const _this = this;
        this.lastResponse = undefined;
        this.lastRequest = new request_1.default(requestOptions);
        return this.axios(options)
            .then((response) => {
            if (opts.logLevel === "debug") {
                console.log(`response.statusCode: ${response.status}`);
                console.log(`response.headers: ${JSON.stringify(response.headers)}`);
            }
            _this.lastResponse = new response_1.default(response.status, response.data, response.headers);
            return {
                statusCode: response.status,
                body: response.data,
                headers: response.headers,
            };
        })
            .catch((error) => {
            _this.lastResponse = undefined;
            throw error;
        });
    }
    filterLoggingHeaders(headers) {
        return Object.keys(headers).filter((header) => {
            return !"authorization".includes(header.toLowerCase());
        });
    }
    logRequest(options) {
        console.log("-- BEGIN Twilio API Request --");
        console.log(`${options.method} ${options.url}`);
        if (options.params) {
            console.log("Querystring:");
            console.log(options.params);
        }
        if (options.headers) {
            console.log("Headers:");
            const filteredHeaderKeys = this.filterLoggingHeaders(options.headers);
            filteredHeaderKeys.forEach((header) => console.log(`${header}: ${options.headers?.header}`));
        }
        console.log("-- END Twilio API Request --");
    }
}
module.exports = RequestClient;
