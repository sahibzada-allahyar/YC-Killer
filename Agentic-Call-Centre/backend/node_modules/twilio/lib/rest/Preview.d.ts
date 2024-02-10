import { FleetListInstance } from "./preview/deployed_devices/fleet";
import { AuthorizationDocumentListInstance } from "./preview/hosted_numbers/authorizationDocument";
import { HostedNumberOrderListInstance } from "./preview/hosted_numbers/hostedNumberOrder";
import { AvailableAddOnListInstance } from "./preview/marketplace/availableAddOn";
import { InstalledAddOnListInstance } from "./preview/marketplace/installedAddOn";
import { ServiceListInstance } from "./preview/sync/service";
import { CommandListInstance } from "./preview/wireless/command";
import { RatePlanListInstance } from "./preview/wireless/ratePlan";
import { SimListInstance } from "./preview/wireless/sim";
import PreviewBase from "./PreviewBase";
declare class Preview extends PreviewBase {
    /**
     * @deprecated - Use deployed_devices.fleets instead
     */
    get fleets(): FleetListInstance;
    /**
     * @deprecated - Use hosted_numbers.authorizationDocuments instead
     */
    get authorizationDocuments(): AuthorizationDocumentListInstance;
    /**
     * @deprecated - Use hosted_numbers.hostedNumberOrders instead
     */
    get hostedNumberOrders(): HostedNumberOrderListInstance;
    /**
     * @deprecated - Use marketplace.availableAddOns instead
     */
    get availableAddOns(): AvailableAddOnListInstance;
    /**
     * @deprecated - Use marketplace.installedAddOns instead
     */
    get installedAddOns(): InstalledAddOnListInstance;
    /**
     * @deprecated - Use sync.services instead
     */
    get services(): ServiceListInstance;
    /**
     * @deprecated - Use wireless.commands instead
     */
    get commands(): CommandListInstance;
    /**
     * @deprecated - Use wireless.ratePlans instead
     */
    get ratePlans(): RatePlanListInstance;
    /**
     * @deprecated - Use wireless.sims instead
     */
    get sims(): SimListInstance;
}
export = Preview;
