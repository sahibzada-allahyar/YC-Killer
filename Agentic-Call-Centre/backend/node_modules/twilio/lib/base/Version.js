"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const RestException_1 = __importDefault(require("./RestException"));
const utility_1 = require("./utility");
class Version {
    /**
     *
     * Base version object
     *
     * @param domain - twilio domain
     * @param version - api version
     */
    constructor(domain, version) {
        this._domain = domain;
        this._version = version;
    }
    get domain() {
        return this._domain;
    }
    /**
     * Generate absolute url from a uri
     *
     * @param uri - uri to transform
     * @returns transformed url
     */
    absoluteUrl(uri) {
        return this._domain.absoluteUrl(this.relativeUrl(uri));
    }
    /**
     * Generate relative url from a uri
     *
     * @param uri - uri to transform
     * @returns transformed url
     */
    relativeUrl(uri) {
        var result = "";
        if (typeof this._version === "string") {
            const version = (0, utility_1.trim)(this._version, "/");
            result += version;
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
     * Make a request against the domain
     *
     * @param opts - request options
     * @returns promise that resolves to request response
     */
    request(opts) {
        return this._domain.request({
            ...opts,
            uri: this.relativeUrl(opts.uri || ""),
        });
    }
    /**
     * Create a new record
     *
     * @param opts - request options
     *
     * @throws Error If response returns non 2xx or 201 status code
     *
     * @returns promise that resolves to created record
     */
    create(opts) {
        var qResponse = this.request(opts);
        qResponse = qResponse.then(function success(response) {
            if (response.statusCode < 200 || response.statusCode >= 300) {
                throw new RestException_1.default(response);
            }
            if (typeof response.body === "string") {
                return JSON.parse(response.body);
            }
            return response.body;
        });
        return qResponse;
    }
    /**
     * Fetch an instance of a record
     *
     * @param opts - request options
     *
     * @throws Error If response returns non 2xx or 3xx status code
     *
     * @returns promise that resolves to fetched result
     */
    fetch(opts) {
        var qResponse = this.request(opts);
        qResponse = qResponse.then(function success(response) {
            if (response.statusCode < 200 || response.statusCode >= 400) {
                throw new RestException_1.default(response);
            }
            if (typeof response.body === "string") {
                return JSON.parse(response.body);
            }
            return response.body;
        });
        return qResponse;
    }
    /**
     * Fetch a page of records
     *
     * @param opts - request options
     * @returns promise that resolves to page of records
     */
    page(opts) {
        return this.request(opts);
    }
    /**
     * Update a record
     *
     * @param opts - request options
     *
     * @throws Error If response returns non 2xx status code
     *
     * @returns promise that resolves to updated result
     */
    update(opts) {
        var qResponse = this.request(opts);
        qResponse = qResponse.then(function success(response) {
            if (response.statusCode < 200 || response.statusCode >= 300) {
                throw new RestException_1.default(response);
            }
            if (typeof response.body === "string") {
                return JSON.parse(response.body);
            }
            return response.body;
        });
        return qResponse;
    }
    /**
     * Delete a record
     *
     * @param opts - request options
     *
     * @throws Error If response returns a 5xx status
     *
     * @returns promise that resolves to true if record was deleted
     */
    remove(opts) {
        var qResponse = this.request(opts);
        qResponse = qResponse.then(function success(response) {
            if (response.statusCode < 200 || response.statusCode >= 300) {
                throw new RestException_1.default(response);
            }
            return response.statusCode === 204;
        });
        return qResponse;
    }
    /**
     * Process limits for list requests
     *
     * @param opts.limit - The maximum number of items to fetch
     * @param opts.pageSize - The maximum number of items to return with every request
     *
     */
    readLimits(opts) {
        var limit = opts.limit;
        var pageSize = opts.pageSize;
        if ((limit && !Number.isFinite(limit)) || limit <= 0) {
            throw new TypeError("Parameter limit must be a positive integer");
        }
        if (pageSize && (!Number.isFinite(pageSize) || pageSize <= 0)) {
            throw new TypeError("Parameter pageSize must be a positive integer");
        }
        if (limit && !pageSize) {
            pageSize = limit;
        }
        return {
            limit: limit,
            pageSize: pageSize,
        };
    }
    setPromiseCallback(operationPromise, callback) {
        if (typeof callback === "function") {
            operationPromise = operationPromise
                .then((value) => callback(null, value))
                .catch((error) => callback(error));
        }
        return operationPromise;
    }
    /**
     * For each record instance, executes a provided callback function with that
     * instance
     *
     * @param params - Parameters (Optional)
     * @param params.limit - Optional maximum number of record instances to
     *  fetch
     * @param params.pageSize - Optional maximum number of records to return
     *  with every request
     * @param params.callback - Callback function to call with each
     *  record instance
     * @param params.done - Optional done function to call when all
     *  records are processed, the limit is reached, or an error occurs.
     *  Receives an error argument if an error occurs.
     * @param callback - Callback function to call with each record.
     *  Receives a done function argument that will short-circuit the for-each
     *  loop that may accept an error argument.
     * @returns Returns a promise that resolves when all records
     *  processed or if the limit is reached, and rejects with an error if an
     *  error occurs and is not handled in the user provided done function.
     */
    each(params, callback) {
        if (typeof params === "function") {
            callback = params;
            params = {};
        }
        else {
            params = params || {};
        }
        if (params.callback) {
            callback = params.callback;
        }
        if (typeof callback === "undefined") {
            throw new Error("Callback function must be provided");
        }
        let done = false;
        let doneCalled = false;
        let currentPage = 1;
        let currentResource = 0;
        let limits = {};
        let pPending = true;
        let pResolve;
        let pReject;
        if (this._version instanceof Version) {
            limits = this._version.readLimits({
                limit: params.limit,
                pageSize: params.pageSize,
            });
        }
        function onComplete(error) {
            let unhandledError = error;
            done = true;
            if (typeof params.done === "function" && !doneCalled) {
                try {
                    params.done(unhandledError);
                    unhandledError = null;
                }
                catch (e) {
                    unhandledError = e;
                }
            }
            doneCalled = true;
            if (pPending) {
                if (unhandledError) {
                    pReject(unhandledError);
                }
                else {
                    pResolve();
                }
                pPending = false;
            }
        }
        function fetchNextPage(fn) {
            let promise = fn();
            if (typeof promise === "undefined") {
                onComplete();
                return;
            }
            promise
                .then((page) => {
                try {
                    page.instances.forEach(function (instance) {
                        if (done ||
                            (typeof params.limit !== "undefined" &&
                                currentResource >= params.limit)) {
                            done = true;
                            return false;
                        }
                        currentResource++;
                        callback?.(instance, onComplete);
                    });
                }
                catch (e) {
                    return onComplete(e);
                }
                if (!done) {
                    currentPage++;
                    fetchNextPage(page.nextPage.bind(page));
                }
                else {
                    onComplete();
                }
            })
                .catch(onComplete);
        }
        return new Promise((resolve, reject) => {
            pResolve = resolve;
            pReject = reject;
            fetchNextPage(this.page.bind(this, Object.assign(params, limits)));
        });
    }
    list(params, callback) {
        if (typeof params === "function") {
            callback = params;
            params = {};
        }
        else {
            params = params || {};
        }
        let allResources = [];
        params.callback = function (resource, done) {
            allResources.push(resource);
            if (typeof params.limit !== "undefined" &&
                allResources.length === params.limit) {
                done();
            }
        };
        let operationPromise = new Promise((resolve, reject) => {
            params.done = function (error) {
                if (typeof error === "undefined") {
                    resolve(allResources);
                }
                else {
                    reject(error);
                }
            };
        });
        if (this._version instanceof Version) {
            operationPromise = this._version.setPromiseCallback(operationPromise, callback);
        }
        this.each(params);
        return operationPromise;
    }
}
exports.default = Version;
