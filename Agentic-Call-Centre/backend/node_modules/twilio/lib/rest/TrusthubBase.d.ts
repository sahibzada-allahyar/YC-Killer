import Domain from "../base/Domain";
import V1 from "./trusthub/V1";
declare class TrusthubBase extends Domain {
    _v1?: V1;
    /**
     * Initialize trusthub domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = TrusthubBase;
