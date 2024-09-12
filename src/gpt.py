import openai

class Gpt4AnswerGenerator:
    def __init__(self, openai_api_key, model='gpt-4o'):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.conversation_history = []

    def get_answer(self, prompt, max_tokens=1024, temperature=0.5):
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            print("An error occurred: ", e)
            return None

        # Extract the assistant's reply from the response
        assistant_response = response.choices[0].message.content

        # Add assistant's reply to the conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })

        return assistant_response
    
    def get_response(self, prompt):
        return self.get_answer(prompt)


#Example
# import os
# api_key = os.getenv('OPENAI_API_KEY')
# generator = Gpt4AnswerGenerator(api_key, model='gpt-4-1106-preview')
# response = generator.get_response('What model are you using?')
# print(response)
