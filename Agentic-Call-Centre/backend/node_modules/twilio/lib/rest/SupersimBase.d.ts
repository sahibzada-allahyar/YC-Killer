import Domain from "../base/Domain";
import V1 from "./supersim/V1";
declare class SupersimBase extends Domain {
    _v1?: V1;
    /**
     * Initialize supersim domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = SupersimBase;
