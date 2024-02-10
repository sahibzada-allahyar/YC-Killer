import ContentBase from "./ContentBase";
import { ContentListInstance } from "./content/v1/content";
declare class Content extends ContentBase {
    /**
     * @deprecated - Use v1.contents instead
     */
    get contents(): ContentListInstance;
}
export = Content;
