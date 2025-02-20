import weaviate
import wikipediaapi
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate.classes.init import AdditionalConfig, Timeout
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import atexit

# Init Weaviate Embedded
clientWeaviate = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    additional_config=AdditionalConfig(
        timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
    )
)

clientWeaviate.connect()
print("DB is ready: {}".format(clientWeaviate.is_ready()))

# Config collection
COLLECTION_NAME = "VietnamCollection"

def create_collection():
    clientWeaviate.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
        properties=[
            Property(
                name="title",
                data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE),
            Property(
                name="content",
                data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE)
        ]
    )

def get_wikipedia_content(title):
    """ Retrieve article content from Wikipedia """
    # Initialize the Wikipedia API
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='user-agent', language='vi')
    page = wiki_wiki.page(title)
    if not page.exists():
        return None
    return page.text

def chunk_text(text):
    """ Chia nội dung thành các phần nhỏ """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # Each chunk has a maximum length of 500 characters
        chunk_overlap=40,   # There is a 40-character overlap between consecutive chunks
    )
    chunks = text_splitter.split_text(text)
    return chunks

def store_in_weaviate(title, content):
    """ Save data to Weaviate """
    collection = clientWeaviate.collections.get(COLLECTION_NAME)
    chunks = chunk_text(content)
    for i, chunk in enumerate(chunks):
        data_object = {
            "title": f"{title} - Part {i+1}",
            "content": chunk
        }
        collection.data.insert(data_object)

def search_weaviate(query, limit=3):
    """ Perform semantic search """
    collection = clientWeaviate.collections.get(COLLECTION_NAME)
    result = collection.query.hybrid(
        query=query,
        query_properties=["content"],
        # alpha = 0 -> Perform keyword-based search  
        # alpha = 1 -> Perform semantic search using embeddings  
        # alpha = 0.5 -> Equal weighting for keyword-based search and semantic search
        alpha=0.5,
        limit=limit
    )
    return result

def keyword_search_weaviate(keyword):
    """ Perform keyword-based search """
    collection = clientWeaviate.collections.get(COLLECTION_NAME)
    result = collection.query.filter({"path": "content", "operator": "Like", "value": f"%{keyword}%"}, limit=10)
    return result

def get_client_openAI():
    clientOpenAI = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="local"
    )
    return clientOpenAI

def init_data(topic):
    # Check and create the collection if it does not exist.

    if clientWeaviate.collections.exists(COLLECTION_NAME):
        print("Collection {} already exists -> delete -> recreate".format(COLLECTION_NAME))
        clientWeaviate.collections.delete(COLLECTION_NAME)
        create_collection()
    else:
        create_collection()

    # Fetch content from Wikipedia
    content = get_wikipedia_content(topic)

    if content:
        store_in_weaviate(topic, content)

def get_promt_context(query):
    # Search
    search_results = search_weaviate(query)
    print("Search Results:", search_results)

    contents = [obj.properties['content'] for obj in search_results.objects]

    CONTEXT = "\n".join(contents)

    prompt = f"""
    Sử dụng CONTEXT sau để trả lời QUESTION ở cuối.
    Nếu bạn không biết câu trả lời, chỉ cần trả lời không biết. Đừng cố bịa câu trả lời.
    Hãy sử dụng tông giọng hài hước.

    CONTEXT: {CONTEXT}

    QUESTION: {query}
    """

    print("----------------------------------------------PROMPT--------------------------------------------------")
    print(prompt)
    print("------------------------------------------------------------------------------------------------")

    return prompt

def end_program():
    # Close the Weaviate connection
    clientWeaviate.close()

init_data("Việt Nam")

# Enter your question here to test
query="Việt Nam có bao nhiêu người"

prompt = get_promt_context(query)

clientOpenAI = get_client_openAI()

response = clientOpenAI.chat.completions.create(
    model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    messages=[
        {"role": "user", "content": prompt},
    ]
)

print(response.choices[0].message.content)

# Register a function to run at program termination
atexit.register(end_program)