import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# RETRIEVAL COMPONENT

def retrieve_relevant_chunks(query, vectorstore, k=3):
    """
    Retrieve most relevant document chunks for a query
    
    Args:
        query: User's question
        vectorstore: Vector database instance
        k: Number of chunks to retrieve
        
    Returns:
        List of relevant document chunks with scores
    """
    print(f"\n[RETRIEVAL] Searching for: '{query}'")
    results = vectorstore.similiarity_search_with_score(query,k=k)
    
    # Format results
    retrieved_chunks=[]
    for doc, score in results:
        retrieved_chunks.append({
            'content': doc.page_content,
            'score': score,
            'metadata': doc.metadata
        })
        
    print(f"Retrieved {len(retrieved_chunks)} relevant chunks")
    return retrieved_chunks

# PROMPT ENGINEERING

def build_rag_prompt(query, retrieved_chunks):
    """
    Construct prompt with context and query
    
    Args:
        query: User's question
        retrieved_chunks: List of retrieeved documents chunks
        
    Returns:
        Formatted prompt sstring
    """
    # combine all retrieve chunks
    context = "\n\n---\n\n".join([chunk['content'] for chunk in retrieved_chunks])
    
    # Build structured prompt
    prompt=f"""
    You are a helpful assistant that answers questions based on the provided context.build_rag_prompt
    
INSTRUCTIONS:
- Use ONLY the information from the context below to answer the question 
- If the answer is not in the context, say "I don't have enough information to answer that question."
- Be concise and accurate
- Cite specific parts of the context when relevant

CONTEXT: {context} 

QUESTION: {query}

ANSWER: 
    """
    
    return prompt

# GENERATION COMPONENT

def generate_response(prompt, model="", temperator=0.3):
    """
    Generate answer using LLM
    
    Args:
        prompt: Complete prompt with context and query
        model: GeminiAI model to use
        temperature: Control randomnext(lower=more factual)
        
    Returns:
        Generated response string
    """
    
    # TODO: implement llm generation...
    
    
# COMPLETE RAG QUERY PIPELINE

def query_rag_system(question, vectorstore, k=3, verbose=True):
    print('RAG QUERY PIPELINE')
    print("="*70)
    
    
    retrieved_chunks=retrieve_relevant_chunks(question, 1)
    
    if verbose:
        print('\n[RETRIEVED CHUNKS]')
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"\nchunk {i} (Score: {chunk['score']:.4f})")
            print(chunk['content'][:200] + '...')
            
    rag_prompt = build_rag_prompt(question, retrieved_chunks)
    
    if verbose:
        print(f"\n[PROMPT]: Length: {rag_prompt}")
        
    answer = generate_response(rag_prompt)
    
    return {
        'question': question,
        'answer': answer,
        'retrieved_chunks': retrieved_chunks,
        'num_chunk_used': len(retrieved_chunks)
    }
    

if __name__ == "__main__":
    print("="*70)
    print("RAG QUERY SYSTEM - Manual Pipeline")
    print("="*70)
    
    # load existing vector store
    # embedding...
    
    vectorstore=Chroma(
        persist_directory="./chroma_db",
        # embedding_function=embedding
    )
    
    print("✅ Vector store loaded")
    
    # Example queries
    queries = [
        'What is this document about?',
        'Tell me about yellow characters',
        'What are the main topicsi covered?'
    ]
    
    for query in queries:
        result = query_rag_system(query, vectorstore, k=3, verbose=True)
    print("\n" + "="*70)
    print("FINAL ANSWER")
    print("="*70)
    
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Chunks used: {result['num_chunks_used']}")
    print("="*70)
    
print("\n✅ RAG system ready for interactive queries")
print("\nTo use interactively")