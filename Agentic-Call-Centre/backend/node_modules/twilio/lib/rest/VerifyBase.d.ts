import Domain from "../base/Domain";
import V2 from "./verify/V2";
declare class VerifyBase extends Domain {
    _v2?: V2;
    /**
     * Initialize verify domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v2(): V2;
}
export = VerifyBase;
