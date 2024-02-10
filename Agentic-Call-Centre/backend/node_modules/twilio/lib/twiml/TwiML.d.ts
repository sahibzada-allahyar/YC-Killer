/**
 * Parent TwiML object
 */
export default class TwiML {
    [key: string]: any;
    response: any;
    _propertyName: string;
    constructor();
    /**
     * Because child elements have properties named after their class names, we need
     * to translate that when we want to access that at the parent prototype level.
     * So this will translate something like Say to 'say' and VoiceResponse to
     * 'response'.
     */
    _getXml(): any;
    /**
     * Convert to TwiML
     *
     * @returns TwiML XML
     */
    toString(): any;
    /**
     * Add text under the TwiML node
     *
     * @param {string} content
     */
    addText(content: string): void;
    /**
     * Add an arbitrary tag as a child.
     *
     * @param {string} content
     */
    addChild(tagName: string, attributes: Object): GenericNode;
}
/**
 * Generic node
 */
declare class GenericNode extends TwiML {
    node: any;
    constructor(node: any);
}
export {};
