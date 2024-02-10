import Domain from "../base/Domain";
import V1 from "./notify/V1";
declare class NotifyBase extends Domain {
    _v1?: V1;
    /**
     * Initialize notify domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = NotifyBase;
