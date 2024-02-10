import { AlertListInstance } from "./monitor/v1/alert";
import { EventListInstance } from "./monitor/v1/event";
import MonitorBase from "./MonitorBase";
declare class Monitor extends MonitorBase {
    /**
     * @deprecated - Use v1.alerts instead
     */
    get alerts(): AlertListInstance;
    /**
     * @deprecated - Use v1.events instead
     */
    get events(): EventListInstance;
}
export = Monitor;
