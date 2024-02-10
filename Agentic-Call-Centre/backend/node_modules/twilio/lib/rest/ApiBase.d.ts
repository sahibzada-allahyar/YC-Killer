import Domain from "../base/Domain";
import V2010 from "./api/V2010";
declare class ApiBase extends Domain {
    _v2010?: V2010;
    /**
     * Initialize api domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v2010(): V2010;
}
export = ApiBase;
