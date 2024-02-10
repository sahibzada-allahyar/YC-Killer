import Domain from "../base/Domain";
import V1 from "./events/V1";
declare class EventsBase extends Domain {
    _v1?: V1;
    /**
     * Initialize events domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = EventsBase;
