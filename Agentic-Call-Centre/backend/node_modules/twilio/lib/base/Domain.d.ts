import { Client as BaseTwilio, RequestOpts } from "./BaseTwilio";
/**
 * Base domain object
 */
export default class Domain {
    twilio: BaseTwilio;
    baseUrl: string;
    /**
     * Creates a Domain instance
     *
     * @param twilio - A Twilio Client
     * @param baseUrl - Base url for this domain
     */
    constructor(twilio: BaseTwilio, baseUrl: string);
    /**
     * Turn a uri into an absolute url
     *
     * @param uri - uri to transform
     * @returns absolute url
     */
    absoluteUrl(uri?: string): string;
    /**
     * Make request to this domain
     *
     * @param opts - request options
     * @returns request promise
     */
    request(opts: RequestOpts): Promise<any>;
}
