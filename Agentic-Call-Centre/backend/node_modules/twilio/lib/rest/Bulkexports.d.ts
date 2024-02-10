import { ExportListInstance } from "./bulkexports/v1/export";
import { ExportConfigurationListInstance } from "./bulkexports/v1/exportConfiguration";
import BulkexportsBase from "./BulkexportsBase";
declare class Bulkexports extends BulkexportsBase {
    /**
     * @deprecated - Use v1.exports instead
     */
    get exports(): ExportListInstance;
    /**
     * @deprecated - Use v1.exportConfiguration instead
     */
    get exportConfiguration(): ExportConfigurationListInstance;
}
export = Bulkexports;
