/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export type ConfigurationStatus = "ok" | "inprogress" | "notstarted";
/**
 * Options to pass to fetch a ConfigurationInstance
 */
export interface ConfigurationContextFetchOptions {
    /** The Pinned UI version of the Configuration resource to fetch. */
    uiVersion?: string;
}
export interface ConfigurationContext {
    /**
     * Fetch a ConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConfigurationInstance
     */
    fetch(callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Fetch a ConfigurationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConfigurationInstance
     */
    fetch(params: ConfigurationContextFetchOptions, callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ConfigurationContextSolution {
}
export declare class ConfigurationContextImpl implements ConfigurationContext {
    protected _version: V1;
    protected _solution: ConfigurationContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    fetch(params?: ConfigurationContextFetchOptions | ((error: Error | null, item?: ConfigurationInstance) => any), callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ConfigurationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ConfigurationResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    attributes: any;
    status: ConfigurationStatus;
    taskrouter_workspace_sid: string;
    taskrouter_target_workflow_sid: string;
    taskrouter_target_taskqueue_sid: string;
    taskrouter_taskqueues: Array<any>;
    taskrouter_skills: Array<any>;
    taskrouter_worker_channels: any;
    taskrouter_worker_attributes: any;
    taskrouter_offline_activity_sid: string;
    runtime_domain: string;
    messaging_service_instance_sid: string;
    chat_service_instance_sid: string;
    flex_service_instance_sid: string;
    flex_instance_sid: string;
    ui_language: string;
    ui_attributes: any;
    ui_dependencies: any;
    ui_version: string;
    service_version: string;
    call_recording_enabled: boolean;
    call_recording_webhook_url: string;
    crm_enabled: boolean;
    crm_type: string;
    crm_callback_url: string;
    crm_fallback_url: string;
    crm_attributes: any;
    public_attributes: any;
    plugin_service_enabled: boolean;
    plugin_service_attributes: any;
    integrations: Array<any>;
    outbound_call_flows: any;
    serverless_service_sids: Array<string>;
    queue_stats_configuration: any;
    notifications: any;
    markdown: any;
    url: string;
    flex_insights_hr: any;
    flex_insights_drilldown: boolean;
    flex_url: string;
    channel_configs: Array<any>;
    debugger_integration: any;
    flex_ui_status_report: any;
    agent_conv_end_methods: any;
    citrix_voice_vdi: any;
    offline_config: any;
}
export declare class ConfigurationInstance {
    protected _version: V1;
    protected _solution: ConfigurationContextSolution;
    protected _context?: ConfigurationContext;
    constructor(_version: V1, payload: ConfigurationResource);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Configuration resource.
     */
    accountSid: string;
    /**
     * The date and time in GMT when the Configuration resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the Configuration resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * An object that contains application-specific data.
     */
    attributes: any;
    status: ConfigurationStatus;
    /**
     * The SID of the TaskRouter Workspace.
     */
    taskrouterWorkspaceSid: string;
    /**
     * The SID of the TaskRouter target Workflow.
     */
    taskrouterTargetWorkflowSid: string;
    /**
     * The SID of the TaskRouter Target TaskQueue.
     */
    taskrouterTargetTaskqueueSid: string;
    /**
     * The list of TaskRouter TaskQueues.
     */
    taskrouterTaskqueues: Array<any>;
    /**
     * The Skill description for TaskRouter workers.
     */
    taskrouterSkills: Array<any>;
    /**
     * The TaskRouter default channel capacities and availability for workers.
     */
    taskrouterWorkerChannels: any;
    /**
     * The TaskRouter Worker attributes.
     */
    taskrouterWorkerAttributes: any;
    /**
     * The TaskRouter SID of the offline activity.
     */
    taskrouterOfflineActivitySid: string;
    /**
     * The URL where the Flex instance is hosted.
     */
    runtimeDomain: string;
    /**
     * The SID of the Messaging service instance.
     */
    messagingServiceInstanceSid: string;
    /**
     * The SID of the chat service this user belongs to.
     */
    chatServiceInstanceSid: string;
    /**
     * The SID of the Flex service instance.
     */
    flexServiceInstanceSid: string;
    /**
     * The SID of the Flex instance.
     */
    flexInstanceSid: string;
    /**
     * The primary language of the Flex UI.
     */
    uiLanguage: string;
    /**
     * The object that describes Flex UI characteristics and settings.
     */
    uiAttributes: any;
    /**
     * The object that defines the NPM packages and versions to be used in Hosted Flex.
     */
    uiDependencies: any;
    /**
     * The Pinned UI version.
     */
    uiVersion: string;
    /**
     * The Flex Service version.
     */
    serviceVersion: string;
    /**
     * Whether call recording is enabled.
     */
    callRecordingEnabled: boolean;
    /**
     * The call recording webhook URL.
     */
    callRecordingWebhookUrl: string;
    /**
     * Whether CRM is present for Flex.
     */
    crmEnabled: boolean;
    /**
     * The CRM type.
     */
    crmType: string;
    /**
     * The CRM Callback URL.
     */
    crmCallbackUrl: string;
    /**
     * The CRM Fallback URL.
     */
    crmFallbackUrl: string;
    /**
     * An object that contains the CRM attributes.
     */
    crmAttributes: any;
    /**
     * The list of public attributes, which are visible to unauthenticated clients.
     */
    publicAttributes: any;
    /**
     * Whether the plugin service enabled.
     */
    pluginServiceEnabled: boolean;
    /**
     * The plugin service attributes.
     */
    pluginServiceAttributes: any;
    /**
     * A list of objects that contain the configurations for the Integrations supported in this configuration.
     */
    integrations: Array<any>;
    /**
     * The list of outbound call flows.
     */
    outboundCallFlows: any;
    /**
     * The list of serverless service SIDs.
     */
    serverlessServiceSids: Array<string>;
    /**
     * Configurable parameters for Queues Statistics.
     */
    queueStatsConfiguration: any;
    /**
     * Configurable parameters for Notifications.
     */
    notifications: any;
    /**
     * Configurable parameters for Markdown.
     */
    markdown: any;
    /**
     * The absolute URL of the Configuration resource.
     */
    url: string;
    /**
     * Object with enabled/disabled flag with list of workspaces.
     */
    flexInsightsHr: any;
    /**
     * Setting this to true will redirect Flex UI to the URL set in flex_url
     */
    flexInsightsDrilldown: boolean;
    /**
     * URL to redirect to in case drilldown is enabled.
     */
    flexUrl: string;
    /**
     * Settings for different limits for Flex Conversations channels attachments.
     */
    channelConfigs: Array<any>;
    /**
     * Configurable parameters for Debugger Integration.
     */
    debuggerIntegration: any;
    /**
     * Configurable parameters for Flex UI Status report.
     */
    flexUiStatusReport: any;
    /**
     * Agent conversation end methods.
     */
    agentConvEndMethods: any;
    /**
     * Citrix voice vdi configuration and settings.
     */
    citrixVoiceVdi: any;
    /**
     * Presence and presence ttl configuration
     */
    offlineConfig: any;
    private get _proxy();
    /**
     * Fetch a ConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConfigurationInstance
     */
    fetch(callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Fetch a ConfigurationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConfigurationInstance
     */
    fetch(params: ConfigurationContextFetchOptions, callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        attributes: any;
        status: ConfigurationStatus;
        taskrouterWorkspaceSid: string;
        taskrouterTargetWorkflowSid: string;
        taskrouterTargetTaskqueueSid: string;
        taskrouterTaskqueues: any[];
        taskrouterSkills: any[];
        taskrouterWorkerChannels: any;
        taskrouterWorkerAttributes: any;
        taskrouterOfflineActivitySid: string;
        runtimeDomain: string;
        messagingServiceInstanceSid: string;
        chatServiceInstanceSid: string;
        flexServiceInstanceSid: string;
        flexInstanceSid: string;
        uiLanguage: string;
        uiAttributes: any;
        uiDependencies: any;
        uiVersion: string;
        serviceVersion: string;
        callRecordingEnabled: boolean;
        callRecordingWebhookUrl: string;
        crmEnabled: boolean;
        crmType: string;
        crmCallbackUrl: string;
        crmFallbackUrl: string;
        crmAttributes: any;
        publicAttributes: any;
        pluginServiceEnabled: boolean;
        pluginServiceAttributes: any;
        integrations: any[];
        outboundCallFlows: any;
        serverlessServiceSids: string[];
        queueStatsConfiguration: any;
        notifications: any;
        markdown: any;
        url: string;
        flexInsightsHr: any;
        flexInsightsDrilldown: boolean;
        flexUrl: string;
        channelConfigs: any[];
        debuggerIntegration: any;
        flexUiStatusReport: any;
        agentConvEndMethods: any;
        citrixVoiceVdi: any;
        offlineConfig: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ConfigurationSolution {
}
export interface ConfigurationListInstance {
    _version: V1;
    _solution: ConfigurationSolution;
    _uri: string;
    (): ConfigurationContext;
    get(): ConfigurationContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ConfigurationListInstance(version: V1): ConfigurationListInstance;
export {};
