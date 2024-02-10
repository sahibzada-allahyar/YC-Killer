import Domain from "../base/Domain";
import V1 from "./video/V1";
declare class VideoBase extends Domain {
    _v1?: V1;
    /**
     * Initialize video domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = VideoBase;
