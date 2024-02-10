import { FormListInstance } from "./verify/v2/form";
import { ServiceListInstance } from "./verify/v2/service";
import { TemplateListInstance } from "./verify/v2/template";
import { VerificationAttemptListInstance } from "./verify/v2/verificationAttempt";
import { VerificationAttemptsSummaryListInstance } from "./verify/v2/verificationAttemptsSummary";
import VerifyBase from "./VerifyBase";
declare class Verify extends VerifyBase {
    /**
     * @deprecated - Use v2.forms instead
     */
    get forms(): FormListInstance;
    /**
     * @deprecated - Use v2.services instead
     */
    get services(): ServiceListInstance;
    /**
     * @deprecated - Use v2.verificationAttempts instead
     */
    get verificationAttempts(): VerificationAttemptListInstance;
    /**
     * @deprecated - Use v2.verificationAttemptsSummary instead
     */
    get verificationAttemptsSummary(): VerificationAttemptsSummaryListInstance;
    /**
     * @deprecated - Use v2.templates instead
     */
    get templates(): TemplateListInstance;
}
export = Verify;
