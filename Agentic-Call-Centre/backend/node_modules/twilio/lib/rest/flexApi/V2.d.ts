import FlexApiBase from "../FlexApiBase";
import Version from "../../base/Version";
import { WebChannelsListInstance } from "./v2/webChannels";
export default class V2 extends Version {
    /**
     * Initialize the V2 version of FlexApi
     *
     * @param domain - The Twilio (Twilio.FlexApi) domain
     */
    constructor(domain: FlexApiBase);
    /** webChannels - { Twilio.FlexApi.V2.WebChannelsListInstance } resource */
    protected _webChannels?: WebChannelsListInstance;
    /** Getter for webChannels resource */
    get webChannels(): WebChannelsListInstance;
}
