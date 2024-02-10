import TrusthubBase from "../TrusthubBase";
import Version from "../../base/Version";
import { ComplianceInquiriesListInstance } from "./v1/complianceInquiries";
import { ComplianceRegistrationInquiriesListInstance } from "./v1/complianceRegistrationInquiries";
import { ComplianceTollfreeInquiriesListInstance } from "./v1/complianceTollfreeInquiries";
import { CustomerProfilesListInstance } from "./v1/customerProfiles";
import { EndUserListInstance } from "./v1/endUser";
import { EndUserTypeListInstance } from "./v1/endUserType";
import { PoliciesListInstance } from "./v1/policies";
import { SupportingDocumentListInstance } from "./v1/supportingDocument";
import { SupportingDocumentTypeListInstance } from "./v1/supportingDocumentType";
import { TrustProductsListInstance } from "./v1/trustProducts";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Trusthub
     *
     * @param domain - The Twilio (Twilio.Trusthub) domain
     */
    constructor(domain: TrusthubBase);
    /** complianceInquiries - { Twilio.Trusthub.V1.ComplianceInquiriesListInstance } resource */
    protected _complianceInquiries?: ComplianceInquiriesListInstance;
    /** complianceRegistrationInquiries - { Twilio.Trusthub.V1.ComplianceRegistrationInquiriesListInstance } resource */
    protected _complianceRegistrationInquiries?: ComplianceRegistrationInquiriesListInstance;
    /** complianceTollfreeInquiries - { Twilio.Trusthub.V1.ComplianceTollfreeInquiriesListInstance } resource */
    protected _complianceTollfreeInquiries?: ComplianceTollfreeInquiriesListInstance;
    /** customerProfiles - { Twilio.Trusthub.V1.CustomerProfilesListInstance } resource */
    protected _customerProfiles?: CustomerProfilesListInstance;
    /** endUsers - { Twilio.Trusthub.V1.EndUserListInstance } resource */
    protected _endUsers?: EndUserListInstance;
    /** endUserTypes - { Twilio.Trusthub.V1.EndUserTypeListInstance } resource */
    protected _endUserTypes?: EndUserTypeListInstance;
    /** policies - { Twilio.Trusthub.V1.PoliciesListInstance } resource */
    protected _policies?: PoliciesListInstance;
    /** supportingDocuments - { Twilio.Trusthub.V1.SupportingDocumentListInstance } resource */
    protected _supportingDocuments?: SupportingDocumentListInstance;
    /** supportingDocumentTypes - { Twilio.Trusthub.V1.SupportingDocumentTypeListInstance } resource */
    protected _supportingDocumentTypes?: SupportingDocumentTypeListInstance;
    /** trustProducts - { Twilio.Trusthub.V1.TrustProductsListInstance } resource */
    protected _trustProducts?: TrustProductsListInstance;
    /** Getter for complianceInquiries resource */
    get complianceInquiries(): ComplianceInquiriesListInstance;
    /** Getter for complianceRegistrationInquiries resource */
    get complianceRegistrationInquiries(): ComplianceRegistrationInquiriesListInstance;
    /** Getter for complianceTollfreeInquiries resource */
    get complianceTollfreeInquiries(): ComplianceTollfreeInquiriesListInstance;
    /** Getter for customerProfiles resource */
    get customerProfiles(): CustomerProfilesListInstance;
    /** Getter for endUsers resource */
    get endUsers(): EndUserListInstance;
    /** Getter for endUserTypes resource */
    get endUserTypes(): EndUserTypeListInstance;
    /** Getter for policies resource */
    get policies(): PoliciesListInstance;
    /** Getter for supportingDocuments resource */
    get supportingDocuments(): SupportingDocumentListInstance;
    /** Getter for supportingDocumentTypes resource */
    get supportingDocumentTypes(): SupportingDocumentTypeListInstance;
    /** Getter for trustProducts resource */
    get trustProducts(): TrustProductsListInstance;
}
