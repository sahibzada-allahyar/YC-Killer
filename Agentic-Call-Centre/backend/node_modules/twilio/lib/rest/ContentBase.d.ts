import Domain from "../base/Domain";
import V1 from "./content/V1";
declare class ContentBase extends Domain {
    _v1?: V1;
    /**
     * Initialize content domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = ContentBase;
