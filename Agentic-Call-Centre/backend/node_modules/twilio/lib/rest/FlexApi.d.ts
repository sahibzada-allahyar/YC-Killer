import { ChannelListInstance } from "./flexApi/v1/channel";
import { ConfigurationListInstance } from "./flexApi/v1/configuration";
import { FlexFlowListInstance } from "./flexApi/v1/flexFlow";
import { InteractionListInstance } from "./flexApi/v1/interaction";
import { WebChannelListInstance } from "./flexApi/v1/webChannel";
import { AssessmentsListInstance } from "./flexApi/v1/assessments";
import { WebChannelsListInstance } from "./flexApi/v2/webChannels";
import FlexApiBase from "./FlexApiBase";
declare class FlexApi extends FlexApiBase {
    /**
     * @deprecated - Use v1.assessments instead
     */
    get assessments(): AssessmentsListInstance;
    /**
     * @deprecated - Use v1.channel instead
     */
    get channel(): ChannelListInstance;
    /**
     * @deprecated - Use v1.configuration instead
     */
    get configuration(): ConfigurationListInstance;
    /**
     * @deprecated - Use v1.flexFlow instead
     */
    get flexFlow(): FlexFlowListInstance;
    /**
     * @deprecated - Use v1.interaction instead
     */
    get interaction(): InteractionListInstance;
    /**
     * @deprecated - Use v1.webChannel instead
     */
    get webChannel(): WebChannelListInstance;
    /**
     * @deprecated - Use v2.webChannels instead
     */
    get webChannels(): WebChannelsListInstance;
}
export = FlexApi;
