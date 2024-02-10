import Domain from "../base/Domain";
import V1 from "./numbers/V1";
import V2 from "./numbers/V2";
declare class NumbersBase extends Domain {
    _v1?: V1;
    _v2?: V2;
    /**
     * Initialize numbers domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
    get v2(): V2;
}
export = NumbersBase;
