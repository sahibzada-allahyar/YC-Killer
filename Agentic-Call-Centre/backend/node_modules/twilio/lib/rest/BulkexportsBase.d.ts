import Domain from "../base/Domain";
import V1 from "./bulkexports/V1";
declare class BulkexportsBase extends Domain {
    _v1?: V1;
    /**
     * Initialize bulkexports domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = BulkexportsBase;
