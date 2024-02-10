import { AccountContext, AccountListInstance } from "./api/v2010/account";
import { AddressListInstance } from "./api/v2010/account/address";
import { ApplicationListInstance } from "./api/v2010/account/application";
import { AuthorizedConnectAppListInstance } from "./api/v2010/account/authorizedConnectApp";
import { AvailablePhoneNumberCountryListInstance } from "./api/v2010/account/availablePhoneNumberCountry";
import { BalanceListInstance } from "./api/v2010/account/balance";
import { CallListInstance } from "./api/v2010/account/call";
import { ConferenceListInstance } from "./api/v2010/account/conference";
import { ConnectAppListInstance } from "./api/v2010/account/connectApp";
import { IncomingPhoneNumberListInstance } from "./api/v2010/account/incomingPhoneNumber";
import { KeyListInstance } from "./api/v2010/account/key";
import { MessageListInstance } from "./api/v2010/account/message";
import { NewKeyListInstance } from "./api/v2010/account/newKey";
import { NewSigningKeyListInstance } from "./api/v2010/account/newSigningKey";
import { NotificationListInstance } from "./api/v2010/account/notification";
import { OutgoingCallerIdListInstance } from "./api/v2010/account/outgoingCallerId";
import { QueueListInstance } from "./api/v2010/account/queue";
import { RecordingListInstance } from "./api/v2010/account/recording";
import { ShortCodeListInstance } from "./api/v2010/account/shortCode";
import { SigningKeyListInstance } from "./api/v2010/account/signingKey";
import { SipListInstance } from "./api/v2010/account/sip";
import { TokenListInstance } from "./api/v2010/account/token";
import { TranscriptionListInstance } from "./api/v2010/account/transcription";
import { UsageListInstance } from "./api/v2010/account/usage";
import { ValidationRequestListInstance } from "./api/v2010/account/validationRequest";
import ApiBase from "./ApiBase";
declare class Api extends ApiBase {
    get account(): AccountContext;
    get accounts(): AccountListInstance;
    /**
     * @deprecated - Use account.addresses instead
     */
    get addresses(): AddressListInstance;
    /**
     * @deprecated - Use account.applications instead
     */
    get applications(): ApplicationListInstance;
    /**
     * @deprecated - Use account.authorizedConnectApps instead
     */
    get authorizedConnectApps(): AuthorizedConnectAppListInstance;
    /**
     * @deprecated - Use account.availablePhoneNumbers instead
     */
    get availablePhoneNumbers(): AvailablePhoneNumberCountryListInstance;
    /**
     * @deprecated - Use account.balance instead
     */
    get balance(): BalanceListInstance;
    /**
     * @deprecated - Use account.calls instead
     */
    get calls(): CallListInstance;
    /**
     * @deprecated - Use account.conferences instead
     */
    get conferences(): ConferenceListInstance;
    /**
     * @deprecated - Use account.connectApps instead
     */
    get connectApps(): ConnectAppListInstance;
    /**
     * @deprecated - Use account.incomingPhoneNumbers instead
     */
    get incomingPhoneNumbers(): IncomingPhoneNumberListInstance;
    /**
     * @deprecated - Use account.keys instead
     */
    get keys(): KeyListInstance;
    /**
     * @deprecated - Use account.messages instead
     */
    get messages(): MessageListInstance;
    /**
     * @deprecated - Use account.newKeys instead
     */
    get newKeys(): NewKeyListInstance;
    /**
     * @deprecated - Use account.newSigningKeys instead
     */
    get newSigningKeys(): NewSigningKeyListInstance;
    /**
     * @deprecated - Use account.notifications instead
     */
    get notifications(): NotificationListInstance;
    /**
     * @deprecated - Use account.outgoingCallerIds instead
     */
    get outgoingCallerIds(): OutgoingCallerIdListInstance;
    /**
     * @deprecated - Use account.queues instead
     */
    get queues(): QueueListInstance;
    /**
     * @deprecated - Use account.recordings instead
     */
    get recordings(): RecordingListInstance;
    /**
     * @deprecated - Use account.signingKeys instead
     */
    get signingKeys(): SigningKeyListInstance;
    /**
     * @deprecated - Use account.sip instead
     */
    get sip(): SipListInstance;
    /**
     * @deprecated - Use account.shortCodes instead
     */
    get shortCodes(): ShortCodeListInstance;
    /**
     * @deprecated - Use account.tokens instead
     */
    get tokens(): TokenListInstance;
    /**
     * @deprecated - Use account.transcriptions instead
     */
    get transcriptions(): TranscriptionListInstance;
    /**
     * @deprecated - Use account.usage instead
     */
    get usage(): UsageListInstance;
    /**
     * @deprecated - Use account.validationRequests instead
     */
    get validationRequests(): ValidationRequestListInstance;
}
export = Api;
