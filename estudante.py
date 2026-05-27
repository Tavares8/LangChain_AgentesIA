from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
import json
import os
from typing import List
import pandas as pd


def busca_dados_de_estudante(estudante):
    dados = pd.read_csv("documentos/estudantes.csv")
    dados["USUARIO"] = dados["USUARIO"].astype(str).str.strip().str.lower()
    dados_com_esse_estudante = dados[dados["USUARIO"] == estudante]
    if dados_com_esse_estudante.empty:
        return {}
    return dados_com_esse_estudante.to_dict(orient="records")[0]

class ExtratorDeEstudante(BaseModel):
    estudante:str = Field(description="Nome do estudante informado, sempre em letras minúsculas.")

class DadosDeEstudante(BaseTool):
    name = "DadosDeEstudante"
    description = """Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico.
Passe para essa ferramenta como argumento o nome do estudante."""

    def _run(self, input: str) -> str:
        llm = ChatOpenAI(model="gpt-4o",
                        api_key=os.getenv("OPENAI_API_KEY")) # executar a ferramento, a LLM precisa ser chamada para extrair o nome do estudante da pergunta, e depois buscar os dados desse estudante. 
        # aqui é outra LLM, diferente da usada em agente.py. São cérebros diferentes, uma para o agente, e outra para a ferramenta. A LLM do agente é responsável por decidir qual ferramenta usar, e a LLM da ferramenta é responsável por executar a tarefa específica daquela ferramenta.
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
        template = PromptTemplate(template="""Você deve analisar a entrada a seguir e extrair o nome informado em minúsculo.
Entrada:
-----------------
{input}
-----------------
                        Formato de saída:
                        {formato_saida}""",
                        input_variables=["input"],
                        partial_variables={"formato_saida" : parser.get_format_instructions()})
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input" : input})
        estudante = resposta['estudante']
        #estudante = input
        estudante = estudante.lower().strip()
        dados = busca_dados_de_estudante(estudante)
        return json.dumps(dados)


class Nota(BaseModel):
    area:str = Field(description="Nome da área de conhecimento")
    nota:float = Field(description="Nota na área de conhecimento")
    
class PerfilAcademicoDeEstudante(BaseModel):
    nome:str = Field(description="nome do estudante")
    ano_de_conclusao:int = Field(description="ano de conclusão")
    notas:List[Nota] = Field(description="Lista de notas das disciplinas e áreas de conhecimento")
    resumo:str = Field(description="Resumo das principais características desse estudante de forma a torná-lo único e um ótimo potencial estudante para faculdades. Exemplo: só este estudante tem bla bla bla")
        
class PerfilAcademico(BaseTool):
    name = "PerfilAcademico"
    description = """Cria um perfil acadêmico de um estudante.
Esta ferramenta requer como entrada todos os dados do estudante.
Eu sou incapaz de buscar os dados do estudante.
Você tem que buscar os dados do estudante antes de me invocar."""

    def _run(self, input:str) -> str:
        llm = ChatOpenAI(model="gpt-4o",
                        api_key=os.getenv("OPENAI_API_KEY"))
        parser = JsonOutputParser(pydantic_object=PerfilAcademicoDeEstudante)
        # Na configuração declarativa uso prompt com arquivos .md. Aqui na programação python, os arquivos .md
        # tornam-se variáveis de texto do tipo ChatPromptTemplate. O template é o que vai guiar a LLM a criar o perfil acadêmico do 
        # estudante, e o parser é o que vai garantir que a resposta da LLM esteja no formato correto para ser convertida em um objeto PerfilAcademicoDeEstudante.
        
        # Em ambas as abordagens visam orquestrar como o LLM decide usar ferramentas externas para realizar tarefas específicas. O template é o guia que orienta a 
        # LLM a criar o perfil acadêmico do estudante, e o parser é o que garante que a resposta da LLM esteja no formato correto para ser convertida em um objeto PerfilAcademicoDeEstudante.
        # Nos dois casos o desafio de engenharia é o mesmo:mitigar alucinações (Estocasticidades) e garantir que o agente siga o fluxo planejado.
        
        template = PromptTemplate(template = """- Formate o estudante para seu perfil acadêmico.
- Com os dados, identifique as opções de universidades sugeridas e cursos compatíveis com o interesse do aluno
- Destaque o perfil do aluno dando enfase principalmente naquilo que faz sentido para as instituições de interesse do aluno

Persona: você é uma consultora de carreira e precisa indicar com detalhes, riqueza, mas direta ao ponto para o estudante as opções e consequências possíveis.
Informações atuais:

{dados_do_estudante}
{formato_de_saida}
""",
            input_variables=["dados_do_estudante"],
            partial_variables={"formato_de_saida" : parser.get_format_instructions()})
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"dados_do_estudante" : input})
        return resposta
        