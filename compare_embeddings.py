from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from setup import initialize_system
import os
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"
load_dotenv()

GOOGLE_API_KEY = initialize_system()
print(GOOGLE_API_KEY, "API key is available")

def main():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", key = GOOGLE_API_KEY)
    vector = embeddings.embed_query("Apple")
    print("vector ",vector)
    print("length of vector",len(vector))
    evaluator = load_evaluator(EvaluatorType.PAIRWISE_EMBEDDING_DISTANCE, embeddings = embeddings)
    words = ["Apple", "Iphone", "fruit"]
    x = [evaluator.evaluate_string_pairs(prediction=word, prediction_b="apple") for word in words]
    for eval, word in zip(x, words):
        print(word, eval)

if __name__ == "__main__":
    main()
