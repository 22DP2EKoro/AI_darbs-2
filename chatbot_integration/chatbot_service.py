import os
from openai import OpenAI
from dotenv import load_dotenv


class ChatbotService:
    def __init__(self):
       # TODO: 1. SOLIS - API atslēgas ielāde 
       # load_dotenv(), lai ielādētu mainīgos no .env faila. # os.getenv(), lai nolasītu "HUGGINGFACE_API_KEY".
        load_dotenv()
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

        if not hf_api_key:
            raise ValueError("HUGGINGFACE_API_KEY nav atrasta .env failā")

       # TODO: 2. SOLIS - OpenAI klienta inicializācija izmantojot "katanemo/Arch-Router-1.5B" modeli
        self.client = OpenAI(
            api_key=hf_api_key,
            base_url="https://api-inference.huggingface.co/v1"
        )

   
        # 3. SOLIS – Sistēmas instrukcijas definēšana
    
        self.system_instruction = (
            "Tu esi noderīgs, draudzīgs un precīzs AI asistents. "
            "Atbildi skaidri un saprotami latviešu valodā."
        )

    def get_chatbot_response(self, user_message, chat_history=None):
        if chat_history is None:
            chat_history = []

       # TODO: 4. SOLIS - Ziņojumu saraksta izveide masīvā # Tajā jābūt sistēmas instrukcijai, visai sarunas vēsture un pēdējai lietotāja ziņa. # 1. Sistēmas instrukcija (role: "system") # 2. Visa iepriekšējā sarunas vēsture (izmantojot .extend(), lai pievienotu visus elementus no chat_history) # 3. Pēdējā lietotāja ziņa (role: "user")
        messages = []

        # 1. Sistēmas instrukcija
        messages.append({
            "role": "system",
            "content": self.system_instruction
        })

        # 2. Iepriekšējā sarunas vēsture
        messages.extend(chat_history)

        # 3. Pēdējā lietotāja ziņa
        messages.append({
            "role": "user",
            "content": user_message
        })

       # TODO: 5. SOLIS - HF API izsaukums ar OpenAI bibliotēku, izmantojot chat.completions.create().
        response = self.client.chat.completions.create(
            model="katanemo/Arch-Router-1.5B",
            messages=messages,
            temperature=0.7,
            max_tokens=512
        )

        # TODO: 6. SOLIS - Atbildes apstrāde un atgriešana # chat.completions.create() atgriež objektu ar "choices" sarakstu, tajā jāparbauda, vai ir pieejama atbilde
        if response.choices and len(response.choices) > 0:
            return {
                "response": response.choices[0].message.content
            }

        return {
            "response": "Netika saņemta atbilde no AI modeļa."
        }
