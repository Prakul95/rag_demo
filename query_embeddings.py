from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from setup import initialize_system
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import argparse

CHROMA_PATH = "chroma"
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    print("Enetring main")
    GOOGLE_API_KEY = initialize_system()
    print(GOOGLE_API_KEY, "API key is available")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", key = GOOGLE_API_KEY)
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    print(query_text)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(results)
    if len(results) == 0 or results[0][1] < 0.4:
        print(f"Unable to find matching results.")
        return
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,
    
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = GOOGLE_API_KEY
    )
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
if __name__=="__main__":
    main()
