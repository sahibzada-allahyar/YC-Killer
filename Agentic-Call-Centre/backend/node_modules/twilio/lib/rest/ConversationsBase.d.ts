import Domain from "../base/Domain";
import V1 from "./conversations/V1";
declare class ConversationsBase extends Domain {
    _v1?: V1;
    /**
     * Initialize conversations domain
     *
     * @param twilio - The twilio client
     */
    constructor(twilio: any);
    get v1(): V1;
}
export = ConversationsBase;
