import Domain from "../base/Domain";
import V1 from "./proxy/V1";
declare class ProxyBase extends Domain {
    _v1?: V1;
    /**
     * Initialize proxy domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = ProxyBase;
