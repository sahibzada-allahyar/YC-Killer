/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
export type FormFormTypes = "form-push";
export interface FormContext {
    /**
     * Fetch a FormInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FormInstance
     */
    fetch(callback?: (error: Error | null, item?: FormInstance) => any): Promise<FormInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface FormContextSolution {
    formType: FormFormTypes;
}
export declare class FormContextImpl implements FormContext {
    protected _version: V2;
    protected _solution: FormContextSolution;
    protected _uri: string;
    constructor(_version: V2, formType: FormFormTypes);
    fetch(callback?: (error: Error | null, item?: FormInstance) => any): Promise<FormInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): FormContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface FormResource {
    form_type: FormFormTypes;
    forms: any;
    form_meta: any;
    url: string;
}
export declare class FormInstance {
    protected _version: V2;
    protected _solution: FormContextSolution;
    protected _context?: FormContext;
    constructor(_version: V2, payload: FormResource, formType?: FormFormTypes);
    formType: FormFormTypes;
    /**
     * Object that contains the available forms for this type. This available forms are given in the standard [JSON Schema](https://json-schema.org/) format
     */
    forms: any;
    /**
     * Additional information for the available forms for this type. E.g. The separator string used for `binding` in a Factor push.
     */
    formMeta: any;
    /**
     * The URL to access the forms for this type.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a FormInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FormInstance
     */
    fetch(callback?: (error: Error | null, item?: FormInstance) => any): Promise<FormInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        formType: "form-push";
        forms: any;
        formMeta: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FormSolution {
}
export interface FormListInstance {
    _version: V2;
    _solution: FormSolution;
    _uri: string;
    (formType: FormFormTypes): FormContext;
    get(formType: FormFormTypes): FormContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FormListInstance(version: V2): FormListInstance;
export {};
