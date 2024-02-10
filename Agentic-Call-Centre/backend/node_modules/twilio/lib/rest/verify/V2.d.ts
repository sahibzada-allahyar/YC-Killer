import VerifyBase from "../VerifyBase";
import Version from "../../base/Version";
import { FormListInstance } from "./v2/form";
import { SafelistListInstance } from "./v2/safelist";
import { ServiceListInstance } from "./v2/service";
import { TemplateListInstance } from "./v2/template";
import { VerificationAttemptListInstance } from "./v2/verificationAttempt";
import { VerificationAttemptsSummaryListInstance } from "./v2/verificationAttemptsSummary";
export default class V2 extends Version {
    /**
     * Initialize the V2 version of Verify
     *
     * @param domain - The Twilio (Twilio.Verify) domain
     */
    constructor(domain: VerifyBase);
    /** forms - { Twilio.Verify.V2.FormListInstance } resource */
    protected _forms?: FormListInstance;
    /** safelist - { Twilio.Verify.V2.SafelistListInstance } resource */
    protected _safelist?: SafelistListInstance;
    /** services - { Twilio.Verify.V2.ServiceListInstance } resource */
    protected _services?: ServiceListInstance;
    /** templates - { Twilio.Verify.V2.TemplateListInstance } resource */
    protected _templates?: TemplateListInstance;
    /** verificationAttempts - { Twilio.Verify.V2.VerificationAttemptListInstance } resource */
    protected _verificationAttempts?: VerificationAttemptListInstance;
    /** verificationAttemptsSummary - { Twilio.Verify.V2.VerificationAttemptsSummaryListInstance } resource */
    protected _verificationAttemptsSummary?: VerificationAttemptsSummaryListInstance;
    /** Getter for forms resource */
    get forms(): FormListInstance;
    /** Getter for safelist resource */
    get safelist(): SafelistListInstance;
    /** Getter for services resource */
    get services(): ServiceListInstance;
    /** Getter for templates resource */
    get templates(): TemplateListInstance;
    /** Getter for verificationAttempts resource */
    get verificationAttempts(): VerificationAttemptListInstance;
    /** Getter for verificationAttemptsSummary resource */
    get verificationAttemptsSummary(): VerificationAttemptsSummaryListInstance;
}
