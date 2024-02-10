import BulkexportsBase from "../BulkexportsBase";
import Version from "../../base/Version";
import { ExportListInstance } from "./v1/export";
import { ExportConfigurationListInstance } from "./v1/exportConfiguration";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of Bulkexports
     *
     * @param domain - The Twilio (Twilio.Bulkexports) domain
     */
    constructor(domain: BulkexportsBase);
    /** exports - { Twilio.Bulkexports.V1.ExportListInstance } resource */
    protected _exports?: ExportListInstance;
    /** exportConfiguration - { Twilio.Bulkexports.V1.ExportConfigurationListInstance } resource */
    protected _exportConfiguration?: ExportConfigurationListInstance;
    /** Getter for exports resource */
    get exports(): ExportListInstance;
    /** Getter for exportConfiguration resource */
    get exportConfiguration(): ExportConfigurationListInstance;
}
