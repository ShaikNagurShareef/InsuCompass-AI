from typing import Optional, Dict, Any, List
from langchain.docstore.document import Document
from insucompass.services import llm_provider
from insucompass.services.vector_store import vector_store_service
from insucompass.core.agents.query_trasformer import QueryTransformationAgent
from insucompass.core.agents.document_summarizer import DocumentSummarizerAgent


llm = llm_provider.get_gemini_llm()
retriever = vector_store_service.get_retriever()
trasformer = QueryTransformationAgent(llm, retriever)
summarizer = DocumentSummarizerAgent(llm_provider.get_llama_llm())

def build_data_doc_dict(docs, summarizer) -> Dict[str, Dict[str, object]]:

    data_doc_dict: Dict[str, Dict[str, object]] = {}
    for idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_id")
        doc_path = doc.metadata.get("source_local_path")

        summary_text: str = summarizer.get_summary(doc_id, doc_path)

        # Store both the raw Document and its summary
        data_doc_dict[doc_id] = {
            "doc": doc,
            "summary": summary_text,
        }

    return data_doc_dict

query = "What is insurance plan? explain the beifits and approaches for enrollment." # decompose
# query = "What is a plan" # step-back
# query = "What is insurance plans are eligible for a person in GA?" # ambiguious

retrieved_docs = trasformer.transform_and_retrieve(query)
summarizer = DocumentSummarizerAgent(llm_provider.get_gemini_fast_llm())

data_doc_dict = build_data_doc_dict(retrieved_docs, summarizer)