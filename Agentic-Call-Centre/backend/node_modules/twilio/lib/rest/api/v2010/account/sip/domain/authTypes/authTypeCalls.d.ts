/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../../../../V2010";
import { AuthCallsCredentialListMappingListInstance } from "./authTypeCalls/authCallsCredentialListMapping";
import { AuthCallsIpAccessControlListMappingListInstance } from "./authTypeCalls/authCallsIpAccessControlListMapping";
export interface AuthTypeCallsSolution {
    accountSid: string;
    domainSid: string;
}
export interface AuthTypeCallsListInstance {
    _version: V2010;
    _solution: AuthTypeCallsSolution;
    _uri: string;
    _credentialListMappings?: AuthCallsCredentialListMappingListInstance;
    credentialListMappings: AuthCallsCredentialListMappingListInstance;
    _ipAccessControlListMappings?: AuthCallsIpAccessControlListMappingListInstance;
    ipAccessControlListMappings: AuthCallsIpAccessControlListMappingListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AuthTypeCallsListInstance(version: V2010, accountSid: string, domainSid: string): AuthTypeCallsListInstance;
