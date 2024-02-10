import ConversationsBase from "../ConversationsBase";
import Version from "../../base/Version";
import { AddressConfigurationListInstance } from "./v1/addressConfiguration";
import { ConfigurationListInstance } from "./v1/configuration";
import { ConversationListInstance } from "./v1/conversation";
import { CredentialListInstance } from "./v1/credential";
import { ParticipantConversationListInstance } from "./v1/participantConversation";
import { RoleListInstance } from "./v1/role";
import { ServiceListInstance } from "./v1/service";
import { UserListInstance } from "./v1/user";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Conversations
     *
     * @param domain - The Twilio (Twilio.Conversations) domain
     */
    constructor(domain: ConversationsBase);
    /** addressConfigurations - { Twilio.Conversations.V1.AddressConfigurationListInstance } resource */
    protected _addressConfigurations?: AddressConfigurationListInstance;
    /** configuration - { Twilio.Conversations.V1.ConfigurationListInstance } resource */
    protected _configuration?: ConfigurationListInstance;
    /** conversations - { Twilio.Conversations.V1.ConversationListInstance } resource */
    protected _conversations?: ConversationListInstance;
    /** credentials - { Twilio.Conversations.V1.CredentialListInstance } resource */
    protected _credentials?: CredentialListInstance;
    /** participantConversations - { Twilio.Conversations.V1.ParticipantConversationListInstance } resource */
    protected _participantConversations?: ParticipantConversationListInstance;
    /** roles - { Twilio.Conversations.V1.RoleListInstance } resource */
    protected _roles?: RoleListInstance;
    /** services - { Twilio.Conversations.V1.ServiceListInstance } resource */
    protected _services?: ServiceListInstance;
    /** users - { Twilio.Conversations.V1.UserListInstance } resource */
    protected _users?: UserListInstance;
    /** Getter for addressConfigurations resource */
    get addressConfigurations(): AddressConfigurationListInstance;
    /** Getter for configuration resource */
    get configuration(): ConfigurationListInstance;
    /** Getter for conversations resource */
    get conversations(): ConversationListInstance;
    /** Getter for credentials resource */
    get credentials(): CredentialListInstance;
    /** Getter for participantConversations resource */
    get participantConversations(): ParticipantConversationListInstance;
    /** Getter for roles resource */
    get roles(): RoleListInstance;
    /** Getter for services resource */
    get services(): ServiceListInstance;
    /** Getter for users resource */
    get users(): UserListInstance;
}
