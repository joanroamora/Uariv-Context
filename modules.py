from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
#from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os


load_dotenv()
#DOCUMENT LOADER

def docLoader(directorio_pdf ):
    # Directorio donde se encuentran los archivos PDF directorio_pdf 
    # Lista para almacenar las páginas de todos los archivos PDF
    pages = []

    # Obtener la lista de archivos PDF en el directorio
    archivos_pdf = os.listdir(directorio_pdf)

    # Recorrer la lista de archivos PDF
    for archivo_pdf in archivos_pdf:

        # Cargar el archivo PDF
        loader = PyPDFLoader(os.path.join(directorio_pdf, archivo_pdf))

        # Dividir el archivo PDF en páginas
        paginas_archivo = loader.load_and_split()

        # Agregar las páginas del archivo actual a la lista global
        pages.extend(paginas_archivo)

    # Imprimir la cantidad de páginas cargadas
    print("DOCUMENT LOADER cargado con Éxito.")
    print(f"Se cargaron un total de {len(pages)} páginas")

    # Retornar la lista de páginas
    return pages

def txt_final (documentsDocumentLoader):
    #variable vacia para el output
    texto_final = ""

    for indice, Document in enumerate(pages):
        # Se agrega el page_content de cada documento al final de la variable texto_final
        texto_final += Document.page_content

        # Si no es el último documento, se agrega un salto de página
        if indice < len(pages) - 1:
            texto_final += "\n\n"

    #   Se imprime el contenido final de la variable texto_final
    print("A continuación los primeros 500 caracteres del texto: " + "\n" + texto_final[:500])
    return texto_final

def textSplitterConfig (chunk_sz , chunk_ol, texto_final ):
    #   Se configura el textsplitter y se generan las particiones
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_sz,
        chunk_overlap=chunk_ol,
        #length_function=len,
        #is_separator_regex=False,
    )   

    texts = text_splitter.split_text(texto_final)
    print(" TEXT SPlITTER aplicado con éxito. Se han creado " +  str(len(texts)) + " documents en total.")
    return texts



def embeddingSetup (model):
    #EMBEDDING SETUP
    embeddings = HuggingFaceEmbeddings(model_name=model)
    print ("Inicialización de Embeddings")
    return embeddings
    

#VECTORSTORE CREATION
def vectorStoreCreation(archivo, embedding):
    knowledge_base = FAISS.from_texts(archivo, embedding)
    print ("VECTORSTORE CREADA CON EXITO ")
    return knowledge_base

#API KEY SETUP langchain open ai chain. LLM selector
def openAIConfig(model_names, chains):
    #os.environ["OPENAI_API_KEY"] = "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"
    llm = ChatOpenAI(model_name=model_names)
    chain = load_qa_chain(llm, chain_type=chains)
    print("Configuración OPENAI lograda con éxito.")
    return llm, chain