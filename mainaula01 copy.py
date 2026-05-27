
from langchain.tools import BaseTool
import os
import json
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
import pandas as pd

load_dotenv()

def busca_dados_de_estudante(estudante):
    print("Buscando dados de estudante: " + estudante)
    dados = pd.read_csv("documentos/estudantes.csv")
    dados["USUARIO"] = dados["USUARIO"].astype(str).str.strip().str.lower()
    dados_com_esse_estudante = dados[dados["USUARIO"] == estudante]
    if dados_com_esse_estudante.empty:
        return {}
    return dados_com_esse_estudante.to_dict(orient="records")[0]

class ExtratorDeEstudante(BaseModel):
    estudante:str = Field(description="Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla.")

class DadosDeEstudante(BaseTool):
    name = "DadosDeEstudante"
    description = """Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico.
Passe para essa ferramenta como argumento o nome do estudante."""

    def _run(self, input: str) -> str:
        llm = ChatOpenAI(model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"))
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
        template = PromptTemplate(template="""Você deve analisar a {input} e extrair o nome de usuário informado.
                        Formato de saída:
                        {formato_saida}""",
                        input_variables=["input"],
                        partial_variables={"formato_saida" : parser.get_format_instructions()})
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input" : input})
        print("\n=== DEBUG: JSON retornado pela LLM ===")
        print(resposta)
        estudante = resposta['estudante']
        estudante = estudante.strip().lower()
        print(estudante)
        dados = busca_dados_de_estudante(estudante) #busca os dados do estudante no csv. 
        print("Achei")
        print(dados)
        return json.dumps(dados)
    
pergunta = "Quais os dados da Carla?"

llm = ChatOpenAI(model="gpt-4o",
                        api_key=os.getenv("OPENAI_API_KEY"))

#resposta = DadosDeEstudante().run(pergunta)
#print(resposta)

dados_de_estudante = DadosDeEstudante()

tools = [
    dados_de_estudante
]

prompt = hub.pull("hwchase17/openai-functions-agent")
input()
print(prompt)
agente = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agente, tools=tools, verbose=True) # vai executar o agente e mostrar o passo a passo
resposta = executor.invoke({"input" : pergunta})
print(resposta) # Ao executar o agente, a LLM vai identificar a ferramenta (classe) que deve ser utilizada, vai chamar a função run da classe DadosDeEstudante, e o resultado dessa função vai ser a resposta final do agente.