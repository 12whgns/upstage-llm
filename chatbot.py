import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
import os
import json


load_dotenv()

# 프로젝트 로깅
logging.langsmith("뷰티하마 QA")

st.title("뷰티하마 QA")

# 처음 1번만 실행
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도
    st.session_state["messages"] = []


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


@st.cache_resource(show_spinner="처리 중입니다...")
def embed_file():
    
    file_path = f"data/bhSenarioV2.pdf"
    folder_path = "vectorstore"

    # 문서 로드
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 임베딩 생성
    embeddings = OpenAIEmbeddings()

    # 벡터스토어가 존재하는지 확인
    if os.path.exists(folder_path):
        # 기존 벡터스토어 로드
        vectorstore = FAISS.load_local( 
            folder_path=folder_path,
            index_name="index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True 
        )
        print("FAISS 벡터 스토어를 로드했습니다.")
    else:
        # 벡터스토어 생성 및 저장
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore.save_local(folder_path=folder_path)
        print("FAISS 벡터 스토어 생성과 저장이 완료되었습니다.")

    # 검색기 생성
    # 문서에 포함되어 있는 정보를 검색 및 생성
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5, "threshold": 0.7}
    )
    return retriever


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # 프롬프트 생성
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # LLM 생성
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 체인 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


retriever = embed_file()
chain = create_chain(retriever, model_name="gpt-4o")
st.session_state["chain"] = chain

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지용 빈 영역
warning_msg = st.empty()

# 사용자 입력시 act
if user_input:
    # chain 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간 생성후 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.error("파일 chainging 실패")
