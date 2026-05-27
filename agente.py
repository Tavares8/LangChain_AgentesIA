from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, create_react_agent
from langchain import hub
import os
from estudante import DadosDeEstudante, PerfilAcademico
from universidade import DadosDeUniversidade, TodasUniversidades

class AgenteOpenAIFunctions:
    def __init__(self):
        llm = ChatOpenAI(model="gpt-4o",
                        api_key=os.getenv("OPENAI_API_KEY")) # o que a LLM vai usar para responder as perguntas.
        # tem as ferramentas disponíveis e toma a decisão de qual usar de acordo com a pergunta. O agente tem acesso a essas ferramentas, e a LLM vai decidir qual usar.

        dados_de_estudante = DadosDeEstudante()
        perfil_academico = PerfilAcademico()
        dados_da_universidade = DadosDeUniversidade()
        todas_universidade = TodasUniversidades()
        self.tools = [
            dados_de_estudante,
            perfil_academico,
            dados_da_universidade,
            todas_universidade
        ]

        # openai functions
        prompt = hub.pull("hwchase17/openai-functions-agent")
        self.agente = create_openai_tools_agent(llm, self.tools, prompt)

        # react
        #prompt = hub.pull("hwchase17/react")
        #self.agente = create_react_agent(llm, self.tools, prompt)