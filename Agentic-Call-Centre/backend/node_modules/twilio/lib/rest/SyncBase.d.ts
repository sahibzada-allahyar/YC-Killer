import Domain from "../base/Domain";
import V1 from "./sync/V1";
declare class SyncBase extends Domain {
    _v1?: V1;
    /**
     * Initialize sync domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = SyncBase;
