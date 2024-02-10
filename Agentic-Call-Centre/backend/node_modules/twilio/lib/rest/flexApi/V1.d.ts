import FlexApiBase from "../FlexApiBase";
import Version from "../../base/Version";
import { AssessmentsListInstance } from "./v1/assessments";
import { ChannelListInstance } from "./v1/channel";
import { ConfigurationListInstance } from "./v1/configuration";
import { FlexFlowListInstance } from "./v1/flexFlow";
import { InsightsAssessmentsCommentListInstance } from "./v1/insightsAssessmentsComment";
import { InsightsConversationsListInstance } from "./v1/insightsConversations";
import { InsightsQuestionnairesListInstance } from "./v1/insightsQuestionnaires";
import { InsightsQuestionnairesCategoryListInstance } from "./v1/insightsQuestionnairesCategory";
import { InsightsQuestionnairesQuestionListInstance } from "./v1/insightsQuestionnairesQuestion";
import { InsightsSegmentsListInstance } from "./v1/insightsSegments";
import { InsightsSessionListInstance } from "./v1/insightsSession";
import { InsightsSettingsAnswerSetsListInstance } from "./v1/insightsSettingsAnswerSets";
import { InsightsSettingsCommentListInstance } from "./v1/insightsSettingsComment";
import { InsightsUserRolesListInstance } from "./v1/insightsUserRoles";
import { InteractionListInstance } from "./v1/interaction";
import { ProvisioningStatusListInstance } from "./v1/provisioningStatus";
import { WebChannelListInstance } from "./v1/webChannel";
export default class V1 extends Version {
    /**
     * Initialize the V1 version of FlexApi
     *
     * @param domain - The Twilio (Twilio.FlexApi) domain
     */
    constructor(domain: FlexApiBase);
    /** assessments - { Twilio.FlexApi.V1.AssessmentsListInstance } resource */
    protected _assessments?: AssessmentsListInstance;
    /** channel - { Twilio.FlexApi.V1.ChannelListInstance } resource */
    protected _channel?: ChannelListInstance;
    /** configuration - { Twilio.FlexApi.V1.ConfigurationListInstance } resource */
    protected _configuration?: ConfigurationListInstance;
    /** flexFlow - { Twilio.FlexApi.V1.FlexFlowListInstance } resource */
    protected _flexFlow?: FlexFlowListInstance;
    /** insightsAssessmentsComment - { Twilio.FlexApi.V1.InsightsAssessmentsCommentListInstance } resource */
    protected _insightsAssessmentsComment?: InsightsAssessmentsCommentListInstance;
    /** insightsConversations - { Twilio.FlexApi.V1.InsightsConversationsListInstance } resource */
    protected _insightsConversations?: InsightsConversationsListInstance;
    /** insightsQuestionnaires - { Twilio.FlexApi.V1.InsightsQuestionnairesListInstance } resource */
    protected _insightsQuestionnaires?: InsightsQuestionnairesListInstance;
    /** insightsQuestionnairesCategory - { Twilio.FlexApi.V1.InsightsQuestionnairesCategoryListInstance } resource */
    protected _insightsQuestionnairesCategory?: InsightsQuestionnairesCategoryListInstance;
    /** insightsQuestionnairesQuestion - { Twilio.FlexApi.V1.InsightsQuestionnairesQuestionListInstance } resource */
    protected _insightsQuestionnairesQuestion?: InsightsQuestionnairesQuestionListInstance;
    /** insightsSegments - { Twilio.FlexApi.V1.InsightsSegmentsListInstance } resource */
    protected _insightsSegments?: InsightsSegmentsListInstance;
    /** insightsSession - { Twilio.FlexApi.V1.InsightsSessionListInstance } resource */
    protected _insightsSession?: InsightsSessionListInstance;
    /** insightsSettingsAnswerSets - { Twilio.FlexApi.V1.InsightsSettingsAnswerSetsListInstance } resource */
    protected _insightsSettingsAnswerSets?: InsightsSettingsAnswerSetsListInstance;
    /** insightsSettingsComment - { Twilio.FlexApi.V1.InsightsSettingsCommentListInstance } resource */
    protected _insightsSettingsComment?: InsightsSettingsCommentListInstance;
    /** insightsUserRoles - { Twilio.FlexApi.V1.InsightsUserRolesListInstance } resource */
    protected _insightsUserRoles?: InsightsUserRolesListInstance;
    /** interaction - { Twilio.FlexApi.V1.InteractionListInstance } resource */
    protected _interaction?: InteractionListInstance;
    /** provisioningStatus - { Twilio.FlexApi.V1.ProvisioningStatusListInstance } resource */
    protected _provisioningStatus?: ProvisioningStatusListInstance;
    /** webChannel - { Twilio.FlexApi.V1.WebChannelListInstance } resource */
    protected _webChannel?: WebChannelListInstance;
    /** Getter for assessments resource */
    get assessments(): AssessmentsListInstance;
    /** Getter for channel resource */
    get channel(): ChannelListInstance;
    /** Getter for configuration resource */
    get configuration(): ConfigurationListInstance;
    /** Getter for flexFlow resource */
    get flexFlow(): FlexFlowListInstance;
    /** Getter for insightsAssessmentsComment resource */
    get insightsAssessmentsComment(): InsightsAssessmentsCommentListInstance;
    /** Getter for insightsConversations resource */
    get insightsConversations(): InsightsConversationsListInstance;
    /** Getter for insightsQuestionnaires resource */
    get insightsQuestionnaires(): InsightsQuestionnairesListInstance;
    /** Getter for insightsQuestionnairesCategory resource */
    get insightsQuestionnairesCategory(): InsightsQuestionnairesCategoryListInstance;
    /** Getter for insightsQuestionnairesQuestion resource */
    get insightsQuestionnairesQuestion(): InsightsQuestionnairesQuestionListInstance;
    /** Getter for insightsSegments resource */
    get insightsSegments(): InsightsSegmentsListInstance;
    /** Getter for insightsSession resource */
    get insightsSession(): InsightsSessionListInstance;
    /** Getter for insightsSettingsAnswerSets resource */
    get insightsSettingsAnswerSets(): InsightsSettingsAnswerSetsListInstance;
    /** Getter for insightsSettingsComment resource */
    get insightsSettingsComment(): InsightsSettingsCommentListInstance;
    /** Getter for insightsUserRoles resource */
    get insightsUserRoles(): InsightsUserRolesListInstance;
    /** Getter for interaction resource */
    get interaction(): InteractionListInstance;
    /** Getter for provisioningStatus resource */
    get provisioningStatus(): ProvisioningStatusListInstance;
    /** Getter for webChannel resource */
    get webChannel(): WebChannelListInstance;
}
