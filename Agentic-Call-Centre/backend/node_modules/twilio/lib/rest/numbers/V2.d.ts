import NumbersBase from "../NumbersBase";
import Version from "../../base/Version";
import { AuthorizationDocumentListInstance } from "./v2/authorizationDocument";
import { BulkHostedNumberOrderListInstance } from "./v2/bulkHostedNumberOrder";
import { HostedNumberOrderListInstance } from "./v2/hostedNumberOrder";
import { RegulatoryComplianceListInstance } from "./v2/regulatoryCompliance";
export default class V2 extends Version {
    /**
     * Initialize the V2 version of Numbers
     *
     * @param domain - The Twilio (Twilio.Numbers) domain
     */
    constructor(domain: NumbersBase);
    /** authorizationDocuments - { Twilio.Numbers.V2.AuthorizationDocumentListInstance } resource */
    protected _authorizationDocuments?: AuthorizationDocumentListInstance;
    /** bulkHostedNumberOrders - { Twilio.Numbers.V2.BulkHostedNumberOrderListInstance } resource */
    protected _bulkHostedNumberOrders?: BulkHostedNumberOrderListInstance;
    /** hostedNumberOrders - { Twilio.Numbers.V2.HostedNumberOrderListInstance } resource */
    protected _hostedNumberOrders?: HostedNumberOrderListInstance;
    /** regulatoryCompliance - { Twilio.Numbers.V2.RegulatoryComplianceListInstance } resource */
    protected _regulatoryCompliance?: RegulatoryComplianceListInstance;
    /** Getter for authorizationDocuments resource */
    get authorizationDocuments(): AuthorizationDocumentListInstance;
    /** Getter for bulkHostedNumberOrders resource */
    get bulkHostedNumberOrders(): BulkHostedNumberOrderListInstance;
    /** Getter for hostedNumberOrders resource */
    get hostedNumberOrders(): HostedNumberOrderListInstance;
    /** Getter for regulatoryCompliance resource */
    get regulatoryCompliance(): RegulatoryComplianceListInstance;
}
