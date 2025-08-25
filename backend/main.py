import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Importações da LangChain para a nova arquitetura de Agente ---
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool # Importação para criar a ferramenta de busca

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Modelo Pydantic para a requisição da API (simplificado, sem histórico)
class ChatRequest(BaseModel):
    message: str

# Configuração da aplicação FastAPI
app = FastAPI(
    title="Assistente Epidemiológico A3Data",
    description="Um assistente conversacional especializado em relatórios epidemiológicos da União Europeia, agora usando uma arquitetura de Agente com Ferramentas.",
)

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name="rag_base",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    return vector_store

vector_store = get_vector_store()

@tool
def search_epidemiological_reports(query: str) -> str:
    """
    Busca informações nos relatórios epidemiológicos da União Europeia de 2021.
    Use esta ferramenta para responder perguntas sobre doenças, estatísticas, dados.
    Args:
        - query str (Deve ser sempre em inglês pois os documentos se encontram em inglês)
    """
    print(f"--- Usando a ferramenta de busca com a query: {query} ---")
    docs = vector_store.similarity_search(query, k=4)
    result = "\n\n".join([doc.page_content for doc in docs])
    return result if result.strip() else "Nenhum documento relevante encontrado."

tools = [search_epidemiological_reports]

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
    max_tokens=1500
)

system_prompt = """Você é um assistente especializado em epidemiologia e vigilância de doenças, desenvolvido pela A3Data para ajudar profissionais de saúde a interpretarem relatórios epidemiológicos da União Europeia de 2021.

Sua função é:
- Analisar e explicar dados epidemiológicos de forma clara e precisa.
- Fornecer insights baseados nos relatórios da UE.
- Manter um tom profissional adequado para médicos e gestores de saúde.
- Ser objetivo e preciso, evitando generalizações.

Diretrizes importantes:
1. Para responder às perguntas, você DEVE usar a ferramenta 'search_epidemiological_reports'.
2. SEMPRE baseie sua resposta final nas informações retornadas pela ferramenta. Não use conhecimento prévio.
3. Se a ferramenta não retornar informações relevantes para a pergunta, informe claramente que não possui essa informação nos documentos disponíveis.
4. Ao fornecer dados numéricos, sempre tente citar a fonte e o ano (2021) se estiverem presentes no contexto recuperado.
5. Mantenha respostas concisas, mas completas o suficiente para serem úteis na prática clínica.
6. Responda sempre em português.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Recebe uma mensagem do usuário
    """
    try:
        response = agent_executor.invoke({
            "input": request.message
        })
        
        return {
            "author": "assistant",
            "content": response["output"]
        }
    except Exception as e:
        return {"error": f"Ocorreu um erro ao processar a solicitação: {str(e)}"}
    
if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY"):
        print("A variável de ambiente OPENAI_API_KEY não foi definida.")
    else:
        print("Iniciando o servidor FastAPI...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
