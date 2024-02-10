import Domain from "../base/Domain";
import V2 from "./intelligence/V2";
declare class IntelligenceBase extends Domain {
    _v2?: V2;
    /**
     * Initialize intelligence domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v2(): V2;
}
export = IntelligenceBase;
