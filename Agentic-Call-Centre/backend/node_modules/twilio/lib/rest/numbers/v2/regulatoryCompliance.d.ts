/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
import { BundleListInstance } from "./regulatoryCompliance/bundle";
import { EndUserListInstance } from "./regulatoryCompliance/endUser";
import { EndUserTypeListInstance } from "./regulatoryCompliance/endUserType";
import { RegulationListInstance } from "./regulatoryCompliance/regulation";
import { SupportingDocumentListInstance } from "./regulatoryCompliance/supportingDocument";
import { SupportingDocumentTypeListInstance } from "./regulatoryCompliance/supportingDocumentType";
export type RegulatoryComplianceEndUserType = "individual" | "business";
export interface RegulatoryComplianceSolution {
}
export interface RegulatoryComplianceListInstance {
    _version: V2;
    _solution: RegulatoryComplianceSolution;
    _uri: string;
    _bundles?: BundleListInstance;
    bundles: BundleListInstance;
    _endUsers?: EndUserListInstance;
    endUsers: EndUserListInstance;
    _endUserTypes?: EndUserTypeListInstance;
    endUserTypes: EndUserTypeListInstance;
    _regulations?: RegulationListInstance;
    regulations: RegulationListInstance;
    _supportingDocuments?: SupportingDocumentListInstance;
    supportingDocuments: SupportingDocumentListInstance;
    _supportingDocumentTypes?: SupportingDocumentTypeListInstance;
    supportingDocumentTypes: SupportingDocumentTypeListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RegulatoryComplianceListInstance(version: V2): RegulatoryComplianceListInstance;
