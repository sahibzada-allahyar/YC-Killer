import Domain from "../base/Domain";
import V1 from "./trunking/V1";
declare class TrunkingBase extends Domain {
    _v1?: V1;
    /**
     * Initialize trunking domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = TrunkingBase;
