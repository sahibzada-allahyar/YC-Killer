/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
import { AwsListInstance } from "./credential/aws";
import { PublicKeyListInstance } from "./credential/publicKey";
export interface CredentialSolution {
}
export interface CredentialListInstance {
    _version: V1;
    _solution: CredentialSolution;
    _uri: string;
    _aws?: AwsListInstance;
    aws: AwsListInstance;
    _publicKey?: PublicKeyListInstance;
    publicKey: PublicKeyListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CredentialListInstance(version: V1): CredentialListInstance;
