import streamlit as st
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

loader = CSVLoader(file_path = 'sales_response.csv')
documents = loader.load()
# print(len(documents))
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


def retrieve_info(query):
    similar_response = db.similarity_search(query, k = 3)
    page_contents_array = [doc.page_content for doc in similar_response]
    print(page_contents_array)
    return page_contents_array





llm = ChatOpenAI(temperature=0.8, model='gpt-4o')

template = """
You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practies, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practies, 
in terms of length, ton of voice, logical arguments and other details

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:
"""



prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template= template
)

chain = LLMChain(llm=llm, prompt=prompt)



def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice = best_practice)
    return response


def main():
    st.set_page_config(
        page_title='Customer response generator', page_icon=":bird:"
    )
    st.header("Customer response generator :bird:")
    message = st.text_area("customer message")
    
# message = """
# Hello,

# I notice your product. I want to know more and send the relevant link to me.

# Best regards,
# Jnchris
# """

# response = generate_response(message)



if __name__ == '__main__':
    main()