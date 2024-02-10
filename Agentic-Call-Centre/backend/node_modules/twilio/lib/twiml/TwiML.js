"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const xmlbuilder_1 = __importDefault(require("xmlbuilder"));
/* jshint ignore:start */
/**
 * Parent TwiML object
 */
/* jshint ignore:end */
class TwiML {
    constructor() {
        this._propertyName = "";
        this.response = xmlbuilder_1.default
            .create("Response", {
            stringify: {
                attValue: function (value) {
                    if (Array.isArray(value)) {
                        value = value.join(" ");
                    }
                    return this.attEscape?.("" + value || "") || "";
                },
            },
        })
            .dec("1.0", "UTF-8");
    }
    /* jshint ignore:start */
    /**
     * Because child elements have properties named after their class names, we need
     * to translate that when we want to access that at the parent prototype level.
     * So this will translate something like Say to 'say' and VoiceResponse to
     * 'response'.
     */
    /* jshint ignore:end */
    _getXml() {
        return this[this._propertyName];
    }
    /* jshint ignore:start */
    /**
     * Convert to TwiML
     *
     * @returns TwiML XML
     */
    /* jshint ignore:end */
    toString() {
        return this._getXml().end();
    }
    /* jshint ignore:start */
    /**
     * Add text under the TwiML node
     *
     * @param {string} content
     */
    /* jshint ignore:end */
    addText(content) {
        this._getXml().txt(content);
    }
    /* jshint ignore:start */
    /**
     * Add an arbitrary tag as a child.
     *
     * @param {string} content
     */
    /* jshint ignore:end */
    addChild(tagName, attributes) {
        return new GenericNode(this._getXml().ele(tagName, attributes));
    }
}
exports.default = TwiML;
/* jshint ignore:start */
/**
 * Generic node
 */
/* jshint ignore:end */
class GenericNode extends TwiML {
    // must match variable name for _getPropertyName
    constructor(node) {
        super();
        this.node = node;
        this._propertyName = "node";
    }
}
