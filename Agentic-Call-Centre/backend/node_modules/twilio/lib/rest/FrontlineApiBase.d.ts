import Domain from "../base/Domain";
import V1 from "./frontlineApi/V1";
declare class FrontlineApiBase extends Domain {
    _v1?: V1;
    /**
     * Initialize frontlineApi domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = FrontlineApiBase;
