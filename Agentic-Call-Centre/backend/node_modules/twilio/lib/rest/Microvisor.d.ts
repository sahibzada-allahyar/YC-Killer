import { AppListInstance } from "./microvisor/v1/app";
import { DeviceListInstance } from "./microvisor/v1/device";
import MicrovisorBase from "./MicrovisorBase";
declare class Microvisor extends MicrovisorBase {
    /**
     * @deprecated - Use v1.apps instead
     */
    get apps(): AppListInstance;
    /**
     * @deprecated - Use v1.devices instead
     */
    get devices(): DeviceListInstance;
}
export = Microvisor;
