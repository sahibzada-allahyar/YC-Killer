import PreviewBase from "../PreviewBase";
import Version from "../../base/Version";
import { AuthorizationDocumentListInstance } from "./hosted_numbers/authorizationDocument";
import { HostedNumberOrderListInstance } from "./hosted_numbers/hostedNumberOrder";
export default class HostedNumbers extends Version {
    /**
     * Initialize the HostedNumbers version of Preview
     *
     * @param domain - The Twilio (Twilio.Preview) domain
     */
    constructor(domain: PreviewBase);
    /** authorizationDocuments - { Twilio.Preview.HostedNumbers.AuthorizationDocumentListInstance } resource */
    protected _authorizationDocuments?: AuthorizationDocumentListInstance;
    /** hostedNumberOrders - { Twilio.Preview.HostedNumbers.HostedNumberOrderListInstance } resource */
    protected _hostedNumberOrders?: HostedNumberOrderListInstance;
    /** Getter for authorizationDocuments resource */
    get authorizationDocuments(): AuthorizationDocumentListInstance;
    /** Getter for hostedNumberOrders resource */
    get hostedNumberOrders(): HostedNumberOrderListInstance;
}
