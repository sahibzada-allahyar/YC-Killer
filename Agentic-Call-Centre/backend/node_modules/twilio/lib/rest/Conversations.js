"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
const ConversationsBase_1 = __importDefault(require("./ConversationsBase"));
class Conversations extends ConversationsBase_1.default {
    /**
     * @deprecated - Use v1.configuration instead
     */
    get configuration() {
        console.warn("configuration is deprecated. Use v1.configuration instead.");
        return this.v1.configuration;
    }
    /**
     * @deprecated - Use v1.addressConfigurations instead
     */
    get addressConfigurations() {
        console.warn("addressConfigurations is deprecated. Use v1.addressConfigurations instead.");
        return this.v1.addressConfigurations;
    }
    /**
     * @deprecated - Use v1.conversations instead
     */
    get conversations() {
        console.warn("conversations is deprecated. Use v1.conversations instead.");
        return this.v1.conversations;
    }
    /**
     * @deprecated - Use v1.credentials instead
     */
    get credentials() {
        console.warn("credentials is deprecated. Use v1.credentials instead.");
        return this.v1.credentials;
    }
    /**
     * @deprecated - Use v1.participantConversations instead
     */
    get participantConversations() {
        console.warn("participantConversations is deprecated. Use v1.participantConversations instead.");
        return this.v1.participantConversations;
    }
    /**
     * @deprecated - Use v1.roles instead
     */
    get roles() {
        console.warn("roles is deprecated. Use v1.roles instead.");
        return this.v1.roles;
    }
    /**
     * @deprecated - Use v1.services instead
     */
    get services() {
        console.warn("services is deprecated. Use v1.services instead.");
        return this.v1.services;
    }
    /**
     * @deprecated - Use v1.users instead
     */
    get users() {
        console.warn("users is deprecated. Use v1.users instead.");
        return this.v1.users;
    }
}
module.exports = Conversations;
