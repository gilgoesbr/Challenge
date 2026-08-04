"""
Módulo principal do agente RAG.
"""
import json
import requests

from retriever import Retriever


class RAGAgent:
    """
    Responsável por orquestrar todo o fluxo RAG.
    """

    def __init__(self):

        self.retriever = Retriever()

    def retrieve_context(self, question: str):

        return self.retriever.search(question)

    def build_context(self, documents):

        context = ""

        for doc in documents:

            context += doc.page_content
            context += "\n\n"

        return context
    def call_ollama(self, context: str, question: str):

    prompt = f"""

Você é um assistente que responde exclusivamente com base no contexto fornecido.

Regras:

- Nunca invente informações.
- Nunca utilize conhecimento externo.
- Se a resposta não estiver no contexto, responda:
  "Não encontrei essa informação nos documentos disponíveis."

Contexto:

{context}

Pergunta:

{question}
"""

    payload = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=120,
    )

    data = response.json()

    return data["response"]

if __name__ == "__main__":

    agent = RAGAgent()

    pergunta = input("Pergunta: ")

    docs = agent.retrieve_context(pergunta)

    contexto = agent.build_context(docs)

    print("=" * 60)
    print(contexto[:1500])        
