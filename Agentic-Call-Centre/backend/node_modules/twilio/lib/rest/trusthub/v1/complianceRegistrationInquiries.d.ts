/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export type ComplianceRegistrationInquiriesBusinessIdentityType = "direct_customer" | "isv_reseller_or_partner" | "unknown";
export type ComplianceRegistrationInquiriesBusinessRegistrationAuthority = "UK:CRN" | "US:EIN" | "CA:CBN" | "AU:ACN" | "Other";
export type ComplianceRegistrationInquiriesEndUserType = "Individual" | "Business";
export type ComplianceRegistrationInquiriesPhoneNumberType = "local" | "national" | "mobile" | "toll-free";
/**
 * Options to pass to create a ComplianceRegistrationInquiriesInstance
 */
export interface ComplianceRegistrationInquiriesListInstanceCreateOptions {
    /**  */
    endUserType: ComplianceRegistrationInquiriesEndUserType;
    /**  */
    phoneNumberType: ComplianceRegistrationInquiriesPhoneNumberType;
    /**  */
    businessIdentityType?: ComplianceRegistrationInquiriesBusinessIdentityType;
    /**  */
    businessRegistrationAuthority?: ComplianceRegistrationInquiriesBusinessRegistrationAuthority;
    /** he name of the business or organization using the Tollfree number. */
    businessLegalName?: string;
    /** he email address to receive the notification about the verification result. */
    notificationEmail?: string;
    /** The email address to receive the notification about the verification result. */
    acceptedNotificationReceipt?: boolean;
    /** Business registration number of the business */
    businessRegistrationNumber?: string;
    /** The URL of the business website */
    businessWebsiteUrl?: string;
    /** Friendly name for your business information */
    friendlyName?: string;
    /** First name of the authorized representative */
    authorizedRepresentative1FirstName?: string;
    /** Last name of the authorized representative */
    authorizedRepresentative1LastName?: string;
    /** Phone number of the authorized representative */
    authorizedRepresentative1Phone?: string;
    /** Email address of the authorized representative */
    authorizedRepresentative1Email?: string;
    /** Birthdate of the authorized representative */
    authorizedRepresentative1DateOfBirth?: string;
    /** Street address of the business */
    addressStreet?: string;
    /** Street address of the business */
    addressStreetSecondary?: string;
    /** City of the business */
    addressCity?: string;
    /** State or province of the business */
    addressSubdivision?: string;
    /** Postal code of the business */
    addressPostalCode?: string;
    /** Country code of the business */
    addressCountryCode?: string;
    /** Street address of the business */
    emergencyAddressStreet?: string;
    /** Street address of the business */
    emergencyAddressStreetSecondary?: string;
    /** City of the business */
    emergencyAddressCity?: string;
    /** State or province of the business */
    emergencyAddressSubdivision?: string;
    /** Postal code of the business */
    emergencyAddressPostalCode?: string;
    /** Country code of the business */
    emergencyAddressCountryCode?: string;
    /** Use the business address as the emergency address */
    useAddressAsEmergencyAddress?: boolean;
    /** The name of the verification document to upload */
    fileName?: string;
    /** The verification document to upload */
    file?: string;
    /** The first name of the Individual User. */
    firstName?: string;
    /** The last name of the Individual User. */
    lastName?: string;
    /** The date of birth of the Individual User. */
    dateOfBirth?: string;
    /** The email address of the Individual User. */
    individualEmail?: string;
    /** The phone number of the Individual User. */
    individualPhone?: string;
    /** Indicates if the inquiry is being started from an ISV embedded component. */
    isIsvEmbed?: boolean;
}
export interface ComplianceRegistrationInquiriesSolution {
}
export interface ComplianceRegistrationInquiriesListInstance {
    _version: V1;
    _solution: ComplianceRegistrationInquiriesSolution;
    _uri: string;
    /**
     * Create a ComplianceRegistrationInquiriesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ComplianceRegistrationInquiriesInstance
     */
    create(params: ComplianceRegistrationInquiriesListInstanceCreateOptions, callback?: (error: Error | null, item?: ComplianceRegistrationInquiriesInstance) => any): Promise<ComplianceRegistrationInquiriesInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ComplianceRegistrationInquiriesListInstance(version: V1): ComplianceRegistrationInquiriesListInstance;
interface ComplianceRegistrationInquiriesResource {
    inquiry_id: string;
    inquiry_session_token: string;
    registration_id: string;
    url: string;
}
export declare class ComplianceRegistrationInquiriesInstance {
    protected _version: V1;
    constructor(_version: V1, payload: ComplianceRegistrationInquiriesResource);
    /**
     * The unique ID used to start an embedded compliance registration session.
     */
    inquiryId: string;
    /**
     * The session token used to start an embedded compliance registration session.
     */
    inquirySessionToken: string;
    /**
     * The RegistrationId matching the Registration Profile that should be resumed or resubmitted for editing.
     */
    registrationId: string;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        inquiryId: string;
        inquirySessionToken: string;
        registrationId: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
