import { CredentialListInstance } from "./chat/v2/credential";
import { ServiceListInstance } from "./chat/v2/service";
import { ChannelListInstance } from "./chat/v3/channel";
import ChatBase from "./ChatBase";
declare class Chat extends ChatBase {
    /**
     * @deprecated - Use v2.credentials instead
     */
    get credentials(): CredentialListInstance;
    /**
     * @deprecated - Use v2.services instead
     */
    get services(): ServiceListInstance;
    /**
     * @deprecated - Use v3.channels instead
     */
    get channels(): ChannelListInstance;
}
export = Chat;
