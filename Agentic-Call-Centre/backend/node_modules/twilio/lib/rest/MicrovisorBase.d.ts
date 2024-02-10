import Domain from "../base/Domain";
import V1 from "./microvisor/V1";
declare class MicrovisorBase extends Domain {
    _v1?: V1;
    /**
     * Initialize microvisor domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = MicrovisorBase;
