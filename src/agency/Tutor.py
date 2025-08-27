# By Romain Puech

import src.agency.Intermediary as Intermediary


class AgencyTutor():

    def __init__(self, client, model="gpt-4o-2024-05-13", intent_history=[], assessment_history=[], open=True, version="V1") -> None:
        #print("---")
        #print("Creating tutor...")
        #print("---")
        self.client = client
        self.model = model#"myGPT4"#model
        self.open = open
        self.intermediary = Intermediary.LLMIntermediary(client=self.client, model=self.model, intent_history=intent_history, assessment_history=assessment_history)

    def update_client(self, client):
        self.client = client
        self.intermediary.update_client(client)

    def update_model(self, model):
        self.model = model
        self.intermediary.update_model(model)

    def get_response(self, conversation_history, messages_student, max_tokens=1500):
        #print("\n---")
        #print("tutor called using model ", self.model)
        prompt, intent, assessment, prompt_tokens, completion_tokens = self.intermediary.get_prompt(conversation_history, messages_student, open=self.open)
        #prompt.append({"role": "system", "content": "Ask the student to find by themself a problem with their answer without giving any hint"})
        #print("prompt generated:")
        #print_logs(prompt)
        

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=max_tokens
        )
        response = completion.choices[0].message.content

        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens
        total_tokens = prompt_tokens + completion_tokens

        response = response.replace("\(","$").replace("\)","$").replace("\[","$$").replace("\]","$$")
        #print("tutor answers:")
        #print(response)
        #print("---")
        return response, total_tokens, prompt_tokens, completion_tokens, intent, assessment
    
    def get_response_stream(self, conversation_history, messages_student, max_tokens=1500):
        #print("\n---")
        #print("tutor called using model ", self.model)
        prompt, intent, assessment, _, _ = self.intermediary.get_prompt(conversation_history, messages_student, open=self.open)
        #prompt.append({"role": "system", "content": "Ask the student to find by themself a problem with their answer without giving any hint"})
        #print("prompt generated:")
        #print_logs(prompt)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=max_tokens,
            stream=True,
        )
        #response = response.replace("\(","$").replace("\)","$").replace("\[","$$").replace("\]","$$")
        #print("tutor answers:")
        #print(response)
        #print("---")
        return stream, intent, assessment