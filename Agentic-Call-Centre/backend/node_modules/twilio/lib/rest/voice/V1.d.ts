import VoiceBase from "../VoiceBase";
import Version from "../../base/Version";
import { ArchivedCallListInstance } from "./v1/archivedCall";
import { ByocTrunkListInstance } from "./v1/byocTrunk";
import { ConnectionPolicyListInstance } from "./v1/connectionPolicy";
import { DialingPermissionsListInstance } from "./v1/dialingPermissions";
import { IpRecordListInstance } from "./v1/ipRecord";
import { SourceIpMappingListInstance } from "./v1/sourceIpMapping";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Voice
     *
     * @param domain - The Twilio (Twilio.Voice) domain
     */
    constructor(domain: VoiceBase);
    /** archivedCalls - { Twilio.Voice.V1.ArchivedCallListInstance } resource */
    protected _archivedCalls?: ArchivedCallListInstance;
    /** byocTrunks - { Twilio.Voice.V1.ByocTrunkListInstance } resource */
    protected _byocTrunks?: ByocTrunkListInstance;
    /** connectionPolicies - { Twilio.Voice.V1.ConnectionPolicyListInstance } resource */
    protected _connectionPolicies?: ConnectionPolicyListInstance;
    /** dialingPermissions - { Twilio.Voice.V1.DialingPermissionsListInstance } resource */
    protected _dialingPermissions?: DialingPermissionsListInstance;
    /** ipRecords - { Twilio.Voice.V1.IpRecordListInstance } resource */
    protected _ipRecords?: IpRecordListInstance;
    /** sourceIpMappings - { Twilio.Voice.V1.SourceIpMappingListInstance } resource */
    protected _sourceIpMappings?: SourceIpMappingListInstance;
    /** Getter for archivedCalls resource */
    get archivedCalls(): ArchivedCallListInstance;
    /** Getter for byocTrunks resource */
    get byocTrunks(): ByocTrunkListInstance;
    /** Getter for connectionPolicies resource */
    get connectionPolicies(): ConnectionPolicyListInstance;
    /** Getter for dialingPermissions resource */
    get dialingPermissions(): DialingPermissionsListInstance;
    /** Getter for ipRecords resource */
    get ipRecords(): IpRecordListInstance;
    /** Getter for sourceIpMappings resource */
    get sourceIpMappings(): SourceIpMappingListInstance;
}
