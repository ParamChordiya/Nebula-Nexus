from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

path = "test_data/test_papers/"
loader = DirectoryLoader(path, glob="**/*.html")
docs = loader.load()

import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)


dataset.to_pandas()