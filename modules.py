from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


#DOCUMENT LOADER
loader = PyPDFLoader("./pdfs/prueba3.pdf")
pages = loader.load_and_split()


#TEXTSPLITTER
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=750,
    chunk_overlap=75,
    #length_function=len,
    #is_separator_regex=False,
)

#FROM DOCUMENTS TO TXT
texto_final = ""

for indice, Document in enumerate(pages):
    # Se agrega el page_content de cada documento al final de la variable texto_final
    texto_final += Document.page_content

    # Si no es el último documento, se agrega un salto de página
    if indice < len(pages) - 1:
        texto_final += "\n\n"

# Se imprime el contenido final de la variable texto_final
print(texto_final)

#SPLITTING OF TXT
texts = text_splitter.split_text(texto_final)

#modelos sugeridos
'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' # 471M
'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' #1.11G


#EMBEDDING SETUP
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

#VECTORSTORE CREATION
knowledge_base = FAISS.from_texts(texts, embeddings)

#PREGUNTA Y EXTRACCION MUESTRAS
pregunta = "¿Cual es el lugar en el que se desarrolla la problematica del texto?"
docs = knowledge_base.similarity_search(pregunta, 3)

#AUTENTICACION OPENAI
os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model_name='gpt-3.5-turbo')

chain = load_qa_chain(llm, chain_type="stuff")

#PREGUNTA
pregunta = "¿Que fue la operacion mariscal en la comuna 13?"
# Busqueda de párrafos similares
docs = knowledge_base.similarity_search(pregunta, 3)
# Utilizar los parrafos similares para darle contexto a ChatGPT
respuesta = chain.run(input_documents=docs, question=pregunta)
print(f"Respuesta ChatGPT: {respuesta}")