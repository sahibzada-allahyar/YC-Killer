/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to fetch a InsightsSettingsAnswerSetsInstance
 */
export interface InsightsSettingsAnswerSetsListInstanceFetchOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
}
export interface InsightsSettingsAnswerSetsSolution {
}
export interface InsightsSettingsAnswerSetsListInstance {
    _version: V1;
    _solution: InsightsSettingsAnswerSetsSolution;
    _uri: string;
    /**
     * Fetch a InsightsSettingsAnswerSetsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsSettingsAnswerSetsInstance
     */
    fetch(callback?: (error: Error | null, item?: InsightsSettingsAnswerSetsInstance) => any): Promise<InsightsSettingsAnswerSetsInstance>;
    /**
     * Fetch a InsightsSettingsAnswerSetsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsSettingsAnswerSetsInstance
     */
    fetch(params: InsightsSettingsAnswerSetsListInstanceFetchOptions, callback?: (error: Error | null, item?: InsightsSettingsAnswerSetsInstance) => any): Promise<InsightsSettingsAnswerSetsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InsightsSettingsAnswerSetsListInstance(version: V1): InsightsSettingsAnswerSetsListInstance;
interface InsightsSettingsAnswerSetsResource {
    account_sid: string;
    answer_sets: any;
    answer_set_categories: any;
    not_applicable: any;
    url: string;
}
export declare class InsightsSettingsAnswerSetsInstance {
    protected _version: V1;
    constructor(_version: V1, payload: InsightsSettingsAnswerSetsResource);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Flex Insights resource and owns this resource.
     */
    accountSid: string;
    /**
     * The lis of answer sets
     */
    answerSets: any;
    /**
     * The list of answer set categories
     */
    answerSetCategories: any;
    /**
     * The details for not applicable answer set
     */
    notApplicable: any;
    url: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        answerSets: any;
        answerSetCategories: any;
        notApplicable: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
