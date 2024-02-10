import Domain from "../base/Domain";
import V1 from "./serverless/V1";
declare class ServerlessBase extends Domain {
    _v1?: V1;
    /**
     * Initialize serverless domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = ServerlessBase;
