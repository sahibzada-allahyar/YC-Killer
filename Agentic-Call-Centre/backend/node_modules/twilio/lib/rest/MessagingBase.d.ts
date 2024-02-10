import Domain from "../base/Domain";
import V1 from "./messaging/V1";
declare class MessagingBase extends Domain {
    _v1?: V1;
    /**
     * Initialize messaging domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = MessagingBase;
