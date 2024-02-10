"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const RestException_1 = __importDefault(require("./RestException"));
class Page {
    /**
     *
     * Base page object to maintain request state.
     *
     * @param version - A twilio version instance
     * @param response - The http response
     * @param solution - path solution
     */
    constructor(version, response, solution) {
        let payload = this.processResponse(response);
        this._version = version;
        this._payload = payload;
        this._solution = solution;
        this.nextPageUrl = this.getNextPageUrl();
        this.previousPageUrl = this.getPreviousPageUrl();
        this.instances = this.loadInstances(this.loadPage(payload));
    }
    /**
     * Get the url of the previous page of records
     *
     * @returns url of the previous page, or undefined if the
     * previous page URI/URL is not defined.
     */
    getPreviousPageUrl() {
        if (this._payload.meta &&
            "previous_page_url" in this._payload.meta &&
            this._payload.meta.previous_page_url) {
            // jshint ignore:line
            return this._payload.meta.previous_page_url; // jshint ignore:line
        }
        if ("previous_page_uri" in this._payload &&
            this._payload.previous_page_uri) {
            // jshint ignore:line
            return this._version._domain.absoluteUrl(this._payload.previous_page_uri); // jshint ignore:line
        }
        return undefined;
    }
    /**
     * Get the url of the next page of records
     *
     * @returns url of the next page, or undefined if the
     * next page URI/URL is not defined.
     */
    getNextPageUrl() {
        if (this._payload.meta &&
            "next_page_url" in this._payload.meta &&
            this._payload.meta.next_page_url) {
            // jshint ignore:line
            return this._payload.meta.next_page_url; // jshint ignore:line
        }
        if ("next_page_uri" in this._payload && this._payload.next_page_uri) {
            // jshint ignore:line
            return this._version._domain.absoluteUrl(this._payload.next_page_uri); // jshint ignore:line
        }
        return undefined;
    }
    /**
     * Build a new instance given a json payload
     *
     * @param payload - Payload response from the API
     * @returns instance of a resource
     */
    getInstance(payload) {
        throw new Error("Page.get_instance() must be implemented in the derived class");
    }
    /**
     * Load a list of records
     *
     * @param resources - json payload of records
     * @returns list of resources
     */
    loadInstances(resources) {
        let instances = [];
        resources.forEach((resource) => {
            instances.push(this.getInstance(resource));
        });
        return instances;
    }
    /**
     * Fetch the next page of records
     *
     * @returns promise that resolves to next page of results,
     * or undefined if there isn't a nextPageUrl undefined.
     */
    nextPage() {
        if (!this.nextPageUrl) {
            return undefined;
        }
        var reqPromise = this._version._domain.twilio.request({
            method: "get",
            uri: this.nextPageUrl,
        });
        var nextPagePromise = reqPromise.then(function (response) {
            return new this.constructor(this._version, response, this._solution);
        }.bind(this));
        return nextPagePromise;
    }
    /**
     * Fetch the previous page of records
     *
     * @returns promise that resolves to previous page of
     * results, or undefined if there isn't a previousPageUrl undefined.
     */
    previousPage() {
        if (!this.previousPageUrl) {
            return undefined;
        }
        var reqPromise = this._version._domain.twilio.request({
            method: "get",
            uri: this.previousPageUrl,
        });
        var prevPagePromise = reqPromise.then(function (response) {
            return new this.constructor(this._version, response, this._solution);
        }.bind(this));
        return prevPagePromise;
    }
    /**
     * Parse json response from API
     *
     * @param response - API response
     *
     * @throws Error If non 200 status code is returned
     *
     * @returns json parsed response
     */
    processResponse(response) {
        if (response.statusCode !== 200) {
            throw new RestException_1.default(response);
        }
        if (typeof response.body === "string") {
            return JSON.parse(response.body);
        }
        return response.body;
    }
    /**
     * Load a page of records
     *
     * @param  {object} payload json payload
     * @return {array} the page of records
     */
    loadPage(payload) {
        if (payload.meta?.key) {
            return payload[payload.meta.key];
        }
        const keys = Object.keys(payload).filter((key) => !Page.META_KEYS.includes(key));
        if (keys.length === 1) {
            return payload[keys[0]];
        }
        throw new Error("Page Records cannot be deserialized");
    }
    forOwn(obj, iteratee) {
        obj = Object(obj);
        for (const [key, val] of Object.entries(obj)) {
            iteratee(val, key, obj);
        }
    }
    toJSON() {
        const clone = {};
        this.forOwn(this, (value, key) => {
            if (!key.startsWith("_") && typeof value !== "function") {
                clone[key] = value;
            }
        });
        return clone;
    }
}
exports.default = Page;
/**
 * Meta keys returned in a list request
 *
 * @constant META_KEYS
 */
Page.META_KEYS = [
    "end",
    "first_page_uri",
    "last_page_uri",
    "next_page_uri",
    "num_pages",
    "page",
    "page_size",
    "previous_page_uri",
    "start",
    "total",
    "uri",
];
