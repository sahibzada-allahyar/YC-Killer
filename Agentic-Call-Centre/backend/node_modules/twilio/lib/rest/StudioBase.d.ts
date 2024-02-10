import Domain from "../base/Domain";
import V1 from "./studio/V1";
import V2 from "./studio/V2";
declare class StudioBase extends Domain {
    _v1?: V1;
    _v2?: V2;
    /**
     * Initialize studio domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
    get v2(): V2;
}
export = StudioBase;
