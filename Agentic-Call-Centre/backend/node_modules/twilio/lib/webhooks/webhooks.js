"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.webhook = exports.validateExpressRequest = exports.validateIncomingRequest = exports.validateRequestWithBody = exports.validateBody = exports.validateRequest = exports.getExpectedBodyHash = exports.getExpectedTwilioSignature = void 0;
const scmp = require("scmp");
const crypto_1 = __importDefault(require("crypto"));
const url_1 = __importDefault(require("url"));
const url_parse_1 = __importDefault(require("url-parse"));
/**
 * Utility function to construct the URL string, since Node.js url library won't include standard port numbers
 *
 * @param parsedUrl - The parsed url object that Twilio requested on your server
 * @returns URL with standard port number included
 */
function buildUrlWithStandardPort(parsedUrl) {
    let url = "";
    const port = parsedUrl.protocol === "https:" ? ":443" : ":80";
    url += parsedUrl.protocol ? parsedUrl.protocol + "//" : "";
    url += parsedUrl.username;
    url += parsedUrl.password ? ":" + parsedUrl.password : "";
    url += parsedUrl.username || parsedUrl.password ? "@" : "";
    url += parsedUrl.host ? parsedUrl.host + port : "";
    url += parsedUrl.pathname + parsedUrl.query + parsedUrl.hash;
    return url;
}
/**
 Utility function to add a port number to a URL

 @param parsedUrl - The parsed url object that Twilio requested on your server
 @returns URL with port
 */
function addPort(parsedUrl) {
    if (!parsedUrl.port) {
        return buildUrlWithStandardPort(parsedUrl);
    }
    return parsedUrl.toString();
}
/**
 Utility function to remove a port number from a URL

 @param parsedUrl - The parsed url object that Twilio requested on your server
 @returns URL without port
 */
function removePort(parsedUrl) {
    parsedUrl.set("port", "");
    return parsedUrl.toString();
}
/**
 Utility function to convert request parameter to a string format

 @param paramName - The request parameter name
 @param paramValue - The request parameter value
 @returns Formatted parameter string
 */
function toFormUrlEncodedParam(paramName, paramValue) {
    if (paramValue instanceof Array) {
        return Array.from(new Set(paramValue))
            .sort()
            .map((val) => toFormUrlEncodedParam(paramName, val))
            .reduce((acc, val) => acc + val, "");
    }
    return paramName + paramValue;
}
/**
 Utility function to get the expected signature for a given request

 @param authToken - The auth token, as seen in the Twilio portal
 @param url - The full URL (with query string) you configured to handle
 this request
 @param params - the parameters sent with this request
 @returns signature
 */
function getExpectedTwilioSignature(authToken, url, params) {
    if (url.indexOf("bodySHA256") !== -1 && params === null) {
        params = {};
    }
    var data = Object.keys(params)
        .sort()
        .reduce((acc, key) => acc + toFormUrlEncodedParam(key, params[key]), url);
    return crypto_1.default
        .createHmac("sha1", authToken)
        .update(Buffer.from(data, "utf-8"))
        .digest("base64");
}
exports.getExpectedTwilioSignature = getExpectedTwilioSignature;
/**
 Utility function to get the expected body hash for a given request's body

 @param body - The plain-text body of the request
 */
function getExpectedBodyHash(body) {
    return crypto_1.default
        .createHash("sha256")
        .update(Buffer.from(body, "utf-8"))
        .digest("hex");
}
exports.getExpectedBodyHash = getExpectedBodyHash;
/**
 Utility function to validate an incoming request is indeed from Twilio

 @param authToken - The auth token, as seen in the Twilio portal
 @param twilioHeader - The value of the X-Twilio-Signature header from the request
 @param url - The full URL (with query string) you configured to handle this request
 @param params - the parameters sent with this request
 @returns valid
 */
function validateRequest(authToken, twilioHeader, url, params) {
    twilioHeader = twilioHeader || "";
    const urlObject = new url_parse_1.default(url);
    const urlWithPort = addPort(urlObject);
    const urlWithoutPort = removePort(urlObject);
    /*
     *  Check signature of the url with and without the port number
     *  since signature generation on the back end is inconsistent
     */
    const signatureWithPort = getExpectedTwilioSignature(authToken, urlWithPort, params);
    const signatureWithoutPort = getExpectedTwilioSignature(authToken, urlWithoutPort, params);
    const validSignatureWithPort = scmp(Buffer.from(twilioHeader), Buffer.from(signatureWithPort));
    const validSignatureWithoutPort = scmp(Buffer.from(twilioHeader), Buffer.from(signatureWithoutPort));
    return validSignatureWithoutPort || validSignatureWithPort;
}
exports.validateRequest = validateRequest;
function validateBody(body, bodyHash) {
    var expectedHash = getExpectedBodyHash(body);
    return scmp(Buffer.from(bodyHash), Buffer.from(expectedHash));
}
exports.validateBody = validateBody;
/**
 Utility function to validate an incoming request is indeed from Twilio. This also validates
 the request body against the bodySHA256 post parameter.

 @param authToken - The auth token, as seen in the Twilio portal
 @param twilioHeader - The value of the X-Twilio-Signature header from the request
 @param url - The full URL (with query string) you configured to handle this request
 @param body - The body of the request
 @returns valid
 */
function validateRequestWithBody(authToken, twilioHeader, url, body) {
    const urlObject = new url_parse_1.default(url, true);
    return (validateRequest(authToken, twilioHeader, url, {}) &&
        validateBody(body, urlObject.query.bodySHA256 || ""));
}
exports.validateRequestWithBody = validateRequestWithBody;
/**
 Utility function to validate an incoming request is indeed from Twilio.
 adapted from https://github.com/crabasa/twiliosig

 @param request - A request object (based on Express implementation http://expressjs.com/api.html#req.params)
 @param authToken - The auth token, as seen in the Twilio portal
 @param opts - options for request validation:
    -> url: The full URL (with query string) you used to configure the webhook with Twilio - overrides host/protocol options
    -> host: manually specify the host name used by Twilio in a number's webhook config
    -> protocol: manually specify the protocol used by Twilio in a number's webhook config
 */
function validateIncomingRequest(request, authToken, opts) {
    var options = opts || {};
    var webhookUrl;
    if (options.url) {
        // Let the user specify the full URL
        webhookUrl = options.url;
    }
    else {
        // Use configured host/protocol, or infer based on request
        var protocol = options.protocol || request.protocol;
        var host = options.host || request.headers.host;
        webhookUrl = url_1.default.format({
            protocol: protocol,
            host: host,
            pathname: request.originalUrl,
        });
        if (request.originalUrl.search(/\?/) >= 0) {
            webhookUrl = webhookUrl.replace(/%3F/g, "?");
        }
    }
    if (webhookUrl.indexOf("bodySHA256") > 0) {
        return validateRequestWithBody(authToken, request.header("X-Twilio-Signature") || "", webhookUrl, request.rawBody || "{}");
    }
    else {
        return validateRequest(authToken, request.header("X-Twilio-Signature") || "", webhookUrl, request.body || {});
    }
}
exports.validateIncomingRequest = validateIncomingRequest;
function validateExpressRequest(request, authToken, opts) {
    return validateIncomingRequest(request, authToken, opts);
}
exports.validateExpressRequest = validateExpressRequest;
/**
Express middleware to accompany a Twilio webhook. Provides Twilio
request validation, and makes the response a little more friendly for our
TwiML generator.  Request validation requires the express.urlencoded middleware
to have been applied (e.g. app.use(express.urlencoded()); in your app config).

Options:
- validate: {Boolean} whether or not the middleware should validate the request
    came from Twilio.  Default true. If the request does not originate from
    Twilio, we will return a text body and a 403.  If there is no configured
    auth token and validate=true, this is an error condition, so we will return
    a 500.
- host: manually specify the host name used by Twilio in a number's webhook config
- protocol: manually specify the protocol used by Twilio in a number's webhook config
- url: The full URL (with query string) you used to configure the webhook with Twilio - overrides host/protocol options

Returns a middleware function.

Examples:
var webhookMiddleware = twilio.webhook();
var webhookMiddleware = twilio.webhook('asdha9dhjasd'); //init with auth token
var webhookMiddleware = twilio.webhook({
    validate:false // don't attempt request validation
});
var webhookMiddleware = twilio.webhook({
    host: 'hook.twilio.com',
    protocol: 'https'
});
 */
function webhook(opts, authToken) {
    let token;
    let options = undefined;
    // Narrowing the args
    if (opts) {
        if (typeof opts === "string") {
            token = opts;
        }
        if (typeof opts === "object") {
            options = opts;
        }
    }
    if (authToken) {
        if (typeof authToken === "string") {
            token = authToken;
        }
        if (typeof authToken === "object") {
            options = authToken;
        }
    }
    if (!options) {
        options = {
            validate: true,
        };
    }
    // Process arguments
    var tokenString;
    for (var i = 0, l = arguments.length; i < l; i++) {
        var arg = arguments[i];
        if (typeof arg === "string") {
            tokenString = arg;
        }
        else {
            options = Object.assign(options || {}, arg);
        }
    }
    // set auth token from input or environment variable
    if (options) {
        options.authToken = tokenString
            ? tokenString
            : process.env.TWILIO_AUTH_TOKEN;
    }
    // Create middleware function
    return function hook(request, response, next) {
        // Do validation if requested
        if (options?.validate) {
            // Check if the 'X-Twilio-Signature' header exists or not
            if (!request.header("X-Twilio-Signature")) {
                return response
                    .type("text/plain")
                    .status(400)
                    .send("No signature header error - X-Twilio-Signature header does not exist, maybe this request is not coming from Twilio.");
            }
            // Check for a valid auth token
            if (!options?.authToken) {
                console.error("[Twilio]: Error - Twilio auth token is required for webhook request validation.");
                response
                    .type("text/plain")
                    .status(500)
                    .send("Webhook Error - we attempted to validate this request without first configuring our auth token.");
            }
            else {
                // Check that the request originated from Twilio
                var valid = validateExpressRequest(request, options?.authToken, {
                    url: options?.url,
                    host: options?.host,
                    protocol: options?.protocol,
                });
                if (valid) {
                    next();
                }
                else {
                    return response
                        .type("text/plain")
                        .status(403)
                        .send("Twilio Request Validation Failed.");
                }
            }
        }
        else {
            next();
        }
    };
}
exports.webhook = webhook;
