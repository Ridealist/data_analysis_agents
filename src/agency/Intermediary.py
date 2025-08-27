# By Romain Puech
import src.agency.prompts.PromptGenerator as PromptGenerator
import src.agency.prompts.TutorActSelector as TutorActSelector
import src.agency.prompts.IntentAssessor as IntentAssessor


class Intermediary():
    def __init__(self, client, model, assessor=None, intentSelector=None, promptGenerator=None, intent_history=[], assessment_history=[]) -> None:
        self.client = client
        self.model = model

        self.assessor = IntentAssessor.IntentAssessor(self.client, self.model, assessment_history=assessment_history) if assessor is None else assessor
        self.intentSelector = TutorActSelector.LLMIntentSelector(intent_history=intent_history) if intentSelector is None else intentSelector
        self.promptGenerator = PromptGenerator.RigidPromptGenerator() if promptGenerator is None else promptGenerator

    def update_client(self, client):
        self.client = client
        self.assessor.client = client
        self.intentSelector.client = client
        self.promptGenerator.client = client
            
    def update_model(self, model):
        self.model = model
        self.assessor.model = model
        self.intentSelector.model = model
        self.promptGenerator.model = model

    def get_prompt(self, conversation_history, student_prompt, open=True):
        #print("generating tutor's prompt...")
        assessment,assessor_prompt_tokens,assessor_completion_tokens = self.assessor.assess(conversation_history, student_prompt)
        intent = self.intentSelector.get_intent(assessment, open=open)
        prompt = self.promptGenerator.get_prompt(conversation_history, student_prompt, intent)
        return prompt, intent, assessment, assessor_prompt_tokens, assessor_completion_tokens


# class EmptyIntermediary(Intermediary):
#     def get_prompt(self, conversation_history, student_prompt, open=True):
#         #print("generating tutor's prompt...")
#         prompt = self.promptGenerator.get_prompt(conversation_history, student_prompt, [])
#         return prompt,[],[],0,0


# class NextStepIntermediary(Intermediary):
#     def __init__(self, client, model, assessor=None, intentSelector=None, promptGenerator=None, intent_history=[], assessment_history=[]) -> None:
#         self.client = client
#         self.model = model

#         self.assessor = IntentAssessor.Assessor(self.client, self.model, assessment_history=assessment_history) if assessor is None else assessor
#         self.intentSelector = TutorActSelector.StrategyIntentSelector(intent_history=intent_history) if intentSelector is None else intentSelector
#         self.promptGenerator = PromptGenerator.RigidPromptGenerator() if promptGenerator is None else promptGenerator


class LLMIntermediary(Intermediary):
    def __init__(self, client, model, assessor=None, intentSelector=None, promptGenerator=None, intent_history=[], assessment_history=[]) -> None:
        self.client = client
        self.model = model
        self.assessor = IntentAssessor.IntentAssessor(self.client,self.model,assessment_history = assessment_history) if assessor is None else assessor
        self.intentSelector = TutorActSelector.LLMIntentSelector(self.client,intent_history=intent_history) if intentSelector is None else intentSelector
        self.promptGenerator = PromptGenerator.RigidPromptGenerator() if promptGenerator is None else promptGenerator


# class FlexibleIntermediary(Intermediary):
#     def __init__(self,client,model,assessor = None, intentSelector=None,promptGenerator = None,intent_history = [],assessment_history=[]) -> None:
#         self.client = client
#         self.model = model

#         self.assessor = IntentAssessor.Assessor(self.client,self.model,assessment_history = assessment_history) if assessor is None else assessor
#         self.intentSelector = TutorActSelector.IntentSelector(intent_history=intent_history) if intentSelector is None else intentSelector
#         self.promptGenerator = PromptGenerator.FlexiblePromptGenerator() if promptGenerator is None else promptGenerator
