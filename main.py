
from modules import docLoader, txt_final, textSplitterConfig, embeddingSetup, vectorStoreCreation, openAIConfig
def main(pregunta):
    pages = docLoader("./pdfs")
    #print (pages)
    texto_final = txt_final(pages)
    texts = textSplitterConfig ( 750 , 75, texto_final)
    
    #modelos sugeridos
    #'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' # 471M
    #'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' #1.11G
    embeddings = embeddingSetup ('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    knowledge_base = vectorStoreCreation (texts, embeddings)
    llm, chain = openAIConfig(model_names="gpt-3.5-turbo", chains="stuff")

    # Busqueda de párrafos similares
    docs = knowledge_base.similarity_search(pregunta, 3)
    # Utilizar los parrafos similares para darle contexto a ChatGPT
    respuesta = chain.run(input_documents=docs, question=pregunta)
    print(f"Respuesta ChatGPT: {respuesta}")

    while True:
        # Solicita al usuario una nueva pregunta
        pregunta = input("Ingrese una nueva pregunta o escriba 'salir' para terminar: ")

        # Verifica si el usuario desea salir
        if pregunta.lower() == 'salir':
            print("Finalizando ejecución.")
            break  # Sale del bucle para terminar la ejecución

        # Busqueda de párrafos similares
        docs = knowledge_base.similarity_search(pregunta, 3)
        # Utilizar los parrafos similares para darle contexto a ChatGPT
        respuesta = chain.run(input_documents=docs, question=pregunta)
        print(f"Respuesta ChatGPT: {respuesta}")



main("¿cuales han sido las principales violaciones a derechos humanos ocurridas en colombia?")