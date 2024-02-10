import Domain from "../base/Domain";
import V1 from "./ipMessaging/V1";
import V2 from "./ipMessaging/V2";
declare class IpMessagingBase extends Domain {
    _v1?: V1;
    _v2?: V2;
    /**
     * Initialize ipMessaging domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
    get v2(): V2;
}
export = IpMessagingBase;
