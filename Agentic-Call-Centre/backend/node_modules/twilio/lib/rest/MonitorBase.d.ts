import Domain from "../base/Domain";
import V1 from "./monitor/V1";
declare class MonitorBase extends Domain {
    _v1?: V1;
    /**
     * Initialize monitor domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = MonitorBase;
