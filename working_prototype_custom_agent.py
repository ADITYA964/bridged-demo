"""# Pinecone query agent"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import Optional, List, Union
from pinecone import Pinecone,delete_index
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.pydantic_v1 import BaseModel, Field, validator

# Load environment variables from .env file
load_dotenv()

# Define the Pydantic model for the desired JSON output
class PineconeFilter(BaseModel):
    author: Optional[str] = Field(None, description="The author's name to filter by.")
    published_day: Optional[Union[int, dict]] = Field(None, description="The publication day, can be an integer or a dictionary with operators like $eq, $gte, $lte.")
    published_year: Optional[Union[int, dict]] = Field(None, description="The publication year, can be an integer or a dictionary with operators like $eq, $gte, $lte.")
    published_month: Optional[Union[int, dict]] = Field(None, description="The publication month (1-12), can be an integer or a dictionary with operators like $eq, $gte, $lte.")
    tags: Optional[dict] = Field(None, description="A dictionary for tags, typically {'$in': ['tag1', 'tag2']}.")
    query: Optional[str] = Field(None, description="The semantic search query part of the natural language input.")

    @validator('published_day','published_year', 'published_month', pre=True)
    def parse_year_month(cls, v):
        if isinstance(v, str):
            try:
                # Try to parse as integer if it's a simple number string
                return int(v)
            except ValueError:
                pass
        return v

class PineconeQueryAgent:
    """
    A Natural Language to Pinecone Query Agent that converts natural language
    inputs into valid Pinecone metadata filters using LangChain's SelfQueryRetriever.
    """

    def __init__(self,
                 index_name="aditya-pinecone-index",
                 llm_model_name="llama3.1",
                 temperature=0,device="cuda",
                 normalize_embeddings=True,
                 hf_embedding_model_name="intfloat/multilingual-e5-large"):

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")

        if not all([self.pinecone_api_key]):
            raise ValueError(
                "Please set PINECONE_API_KEY "
                "environment variables or in a .env file."
            )

        # Initialize LLM and Embeddings
        self.llm = ChatOllama(model=llm_model_name,temperature=temperature)

        self.model_kwargs = {'device': device}

        self.encode_kwargs = {'normalize_embeddings': normalize_embeddings}

        self.embeddings = HuggingFaceEmbeddings(model_name=hf_embedding_model_name,
                                                model_kwargs=self.model_kwargs,
                                                encode_kwargs=self.encode_kwargs)

        # Define the metadata schema for the SelfQueryRetriever
        self.document_content_description = "Articles, blog posts, and research papers"

        self.metadata_field_info = [
            AttributeInfo(
                name="published_day",
                description="The day the article was published (e.g., 1)",
                type="integer",
            ),
            AttributeInfo(
                name="published_year",
                description="The year the article was published (e.g., 2023)",
                type="integer",
            ),
            AttributeInfo(
                name="published_month",
                description="The month the article was published (1-12, e.g., 6 for June)",
                type="integer",
            ),
            AttributeInfo(
                name="author",
                description="The name of the author of the article (e.g., Alice Zhang)",
                type="string",
            ),
            AttributeInfo(
                name="tags",
                description="A list of topics or categories the article is about (e.g., ['machine learning', 'LLMs'])",
                type="list[string]",
            ),
        ]

        # # Initialize Pinecone Vector Store
        self.index_name = index_name # Use a consistent index name

        self.vectorstore = None

        self.pc = Pinecone(api_key=self.pinecone_api_key)

        if not self.pc.has_index(self.index_name):

          self.pc.create_index_for_model(
                  name=self.index_name,
                  cloud="aws",
                  region="us-east-1",
                  embed={
                      "model":"multilingual-e5-large",
                      "field_map":{"text": "title"}
                  }
              )
          print(f"Pinecone index '{self.index_name}' is generated.")
        else:
          print(f"Pinecone index '{self.index_name}' already exists.")


        # Initialize the SelfQueryRetriever
        self.retriever = None

    def _initialize_pinecone_vectorstore(self,index_name,documents):

      self.vectorstore = PineconeVectorStore.from_texts([doc[0] for doc in documents],
                                                        self.embeddings,
                                                        index_name=index_name,
                                                        metadatas=[doc[1] for doc in documents])

    def _delete_pinecone_index(self,index_name):

      self.pc.delete_index(index_name)

      print(f"Pinecone index '{index_name}' is successfully deleted.")

    def _create_self_retriever(self,filter):

      self.retriever = SelfQueryRetriever.from_llm(self.llm,
                                                   self.vectorstore,
                                                   self.document_content_description,
                                                   self.metadata_field_info,
                                                   verbose=False,
                                                   search_kwargs={
                                                    'filter': filter})

    def _inference(self,nl_query,parser):
      self.prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant that extracts Pinecone metadata filters and a semantic query from natural language input. "
                           "The available metadata fields are: 'author' (string), 'published_year' (integer), 'published_month' (integer), and 'tags' (list of strings). "
                           "For dates, use 'published_year' ,'published_day'and 'published_month' as integers. "
                           "For tags, use the format: {{'tags': {{'$in': ['tag1', 'tag2']}}}}. "
                           "If no specific filter is mentioned, return an empty dictionary for that field. "
                           "Extract the main semantic search part of the query into the 'query' field. "
                           "Output only a JSON object following this schema:\n{format_instructions}"),
                            ("user", "{query}")]).partial(format_instructions=parser.get_format_instructions())

      self.chain = self.prompt | self.llm | parser

      # Invoke the chain to get the structured output
      self.structured_output = self.chain.invoke({"query": nl_query})

       # Construct the final Pinecone filter based on the structured output
      self.final_filter = {}
      if self.structured_output.get("author"):
          self.final_filter["author"] = self.structured_output["author"]
      if self.structured_output.get("published_year") is not None:
          # Handle cases where LLM might return an int or a dict for year/month
          if isinstance(self.structured_output["published_year"], int):
              self.final_filter["published_year"] = {"$eq": self.structured_output["published_year"]}
          elif isinstance(self.structured_output["published_year"], dict):
              self.final_filter["published_year"] = self.structured_output["published_year"]
      if self.structured_output.get("published_month") is not None:
          if isinstance(self.structured_output["published_month"], int):
              self.final_filter["published_month"] = {"$eq": self.structured_output["published_month"]}
          elif isinstance(self.structured_output["published_month"], dict):
              self.final_filter["published_month"] = self.structured_output["published_month"]
      if self.structured_output.get("published_day") is not None:
          if isinstance(self.structured_output["published_day"], int):
              self.final_filter["published_day"] = {"$eq": self.structured_output["published_day"]}
          elif isinstance(self.structured_output["published_day"], dict):
              self.final_filter["published_day"] = self.structured_output["published_day"]
      if self.structured_output.get("tags"):
          self.final_filter["tags"] = self.structured_output["tags"]

      return self.final_filter

"""# Preprocessing dataset"""

import pandas as pd

url = "https://docs.google.com/spreadsheets/d/1yky4n9AtCms7cniQ3CahdaaBOpt0gEWcl2VcJHMvMQ8/export?format=csv&gid=2005119392"

df = pd.read_csv(url)

df['publishedDate'] = pd.to_datetime(df['publishedDate'])
df['published_year'] = df['publishedDate'].dt.year
df['published_month'] = df['publishedDate'].dt.month
df['published_day'] = df['publishedDate'].dt.day

import ast

# Safely convert tags string to list
df['tags'] = df['tags'].apply(ast.literal_eval)

# Convert to list of documents
documents = [
    (
        row['title'],
        {
            'author': row['author'],
            'published_year': int(row['published_year']),
            'published_month': int(row['published_month']),
            'published_day': int(row['published_day']),
            'tags': row['tags']
        }
    )
    for _, row in df.iterrows()
]

# Optional: Print or use the documents
for doc in documents:
    print(doc)

"""# Initialize custom agent"""

agent = PineconeQueryAgent()

agent._initialize_pinecone_vectorstore('aditya-pinecone-index',documents)

llm_output_parser = JsonOutputParser(pydantic_object=PineconeFilter)

"""# Natural language query as input to custom agent"""

natural_lang_query = "Anything by Jane Doe on #MumbaiIndians on 1 May 2025?"

pinecone_query = agent._inference(natural_lang_query,llm_output_parser)

"""# Custom Agent output"""

pinecone_query

"""# Using vector search and metadata filtering"""

agent._create_self_retriever(pinecone_query)

natural_lang_query

agent.retriever.invoke(natural_lang_query)

"""# Delete Pinecone index after using."""

agent._delete_pinecone_index('aditya-pinecone-index')