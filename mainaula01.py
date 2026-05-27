from langchain.agents import  AgentExecutor
from agente import AgenteOpenAIFunctions
from dotenv import load_dotenv

load_dotenv()
    
pergunta = "Quais os dados da Ana?"
pergunta = "Quais os dados da Bianca?"
pergunta = "Quais os dados da Ana e da Bianca?"

agente = AgenteOpenAIFunctions()      
executor = AgentExecutor(agent=agente.agente, tools=agente.tools, verbose=True) # vai executar o agente e mostrar o passo a passo
resposta = executor.invoke({"input" : pergunta})
print(resposta) # Ao executar o agente, a LLM vai 
#identificar a ferramenta (classe) que deve ser utilizada, vai chamar a função run da classe DadosDeEstudante, e o resultado dessa função vai ser a resposta final do agente.