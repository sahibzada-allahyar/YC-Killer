import Domain from "../base/Domain";
import V1 from "./pricing/V1";
import V2 from "./pricing/V2";
declare class PricingBase extends Domain {
    _v1?: V1;
    _v2?: V2;
    /**
     * Initialize pricing domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
    get v2(): V2;
}
export = PricingBase;
