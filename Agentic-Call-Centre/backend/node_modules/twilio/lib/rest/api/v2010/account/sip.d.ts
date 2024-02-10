/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../V2010";
import { CredentialListListInstance } from "./sip/credentialList";
import { DomainListInstance } from "./sip/domain";
import { IpAccessControlListListInstance } from "./sip/ipAccessControlList";
export interface SipSolution {
    accountSid: string;
}
export interface SipListInstance {
    _version: V2010;
    _solution: SipSolution;
    _uri: string;
    _credentialLists?: CredentialListListInstance;
    credentialLists: CredentialListListInstance;
    _domains?: DomainListInstance;
    domains: DomainListInstance;
    _ipAccessControlLists?: IpAccessControlListListInstance;
    ipAccessControlLists: IpAccessControlListListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SipListInstance(version: V2010, accountSid: string): SipListInstance;
