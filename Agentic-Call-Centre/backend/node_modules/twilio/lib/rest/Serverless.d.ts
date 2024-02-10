import { ServiceListInstance } from "./serverless/v1/service";
import ServerlessBase from "./ServerlessBase";
declare class Serverless extends ServerlessBase {
    /**
     * @deprecated - Use v1.services instead
     */
    get services(): ServiceListInstance;
}
export = Serverless;
