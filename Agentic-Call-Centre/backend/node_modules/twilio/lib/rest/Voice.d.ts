import { ArchivedCallListInstance } from "./voice/v1/archivedCall";
import { ByocTrunkListInstance } from "./voice/v1/byocTrunk";
import { ConnectionPolicyListInstance } from "./voice/v1/connectionPolicy";
import { DialingPermissionsListInstance } from "./voice/v1/dialingPermissions";
import { IpRecordListInstance } from "./voice/v1/ipRecord";
import { SourceIpMappingListInstance } from "./voice/v1/sourceIpMapping";
import VoiceBase from "./VoiceBase";
declare class Voice extends VoiceBase {
    /**
     * @deprecated - Use v1.archivedCalls instead
     */
    get archivedCalls(): ArchivedCallListInstance;
    /**
     * @deprecated - Use v1.byocTrunks instead
     */
    get byocTrunks(): ByocTrunkListInstance;
    /**
     * @deprecated - Use v1.connectionPolicies instead
     */
    get connectionPolicies(): ConnectionPolicyListInstance;
    /**
     * @deprecated - Use v1.dialingPermissions instead
     */
    get dialingPermissions(): DialingPermissionsListInstance;
    /**
     * @deprecated - Use v1.ipRecords instead
     */
    get ipRecords(): IpRecordListInstance;
    /**
     * @deprecated - Use v1.sourceIpMappings instead
     */
    get sourceIpMappings(): SourceIpMappingListInstance;
}
export = Voice;
