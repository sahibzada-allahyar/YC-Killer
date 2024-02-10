import ChatBase from "../ChatBase";
import Version from "../../base/Version";
import { ChannelListInstance } from "./v3/channel";
export default class V3 extends Version {
    /**
     * Initialize the V3 version of Chat
     *
     * @param domain - The Twilio (Twilio.Chat) domain
     */
    constructor(domain: ChatBase);
    /** channels - { Twilio.Chat.V3.ChannelListInstance } resource */
    protected _channels?: ChannelListInstance;
    /** Getter for channels resource */
    get channels(): ChannelListInstance;
}
