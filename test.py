#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import numpy as np
import sys
import time
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from colorama import Fore, Back, Style
MAX_CONTEXT_QUESTIONS = 2
####This code will remember past conversation.Currenlty we have saved past 3 responses because of rate limit issue
################################################################################
### Step 1
################################################################################                                                                                                                                                                                                   

#load_dotenv(Path(r"C:\Users\norin.saiyed\web-crawl-q-and-a\.env"))
load_dotenv()
# api_key = os.environ["API_KEY"]
# openai.api_key = api_key
openai.api_key = '   '
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["Company_specification"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question_2(
    df,
    model="gpt-3.5-turbo",
    #text-davinci-003
    question="",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=1800,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    #print(context)
    # If debug, print the raw model response
    if debug:
        
        print("\n\n")

    try:
        prompt= f"Question: {question}\nAnswer:",

        instructions = f"""You are a conversational chatbot assistant:Regardless of the sequence, if a user asks questions from either AI Hub, Clean Energy, packaging division or the Recommendation Bot, you will provide the correct answers in their given formats only and maintain a friendly tone.
        Ai hub
        - Provide a list of companies in consulting, manufacturing, hardware, or infrastructure verticals.
        - Ask for the user's specific preference if they're undecided.
        - Respond exclusively about consulting, manufacturing, hardware, or infrastructure within the Ai hub division.
        - Implement error handling for unclear queries or queries outside the context.
        - Conclude each response with open-ended questions based on the response for further discussion.
        
        Clean Energy Instructions:
        - Provide a list of companies in wind, hydropower, nuclear, geothermal, and solar verticals.
        - Ask for the user's specific preference if they're undecided.
        - Respond exclusively about wind, hydropower, nuclear, geothermal, and solar panels within the Clean Energy division.
        - Implement error handling for unclear queries or queries outside the context.
        - Conclude each response with open-ended questions based on the response for further discussion.
        
        Packaging Instructions:
        - Provide a list of available verticals in glass, plastic, metal, or paper.
        - Ask for the user's specific preference if they're undecided.
        - Implement error handling for off-topic queries or when there are no more relevant companies in the division.
        - Conclude each response with open-ended questions based on the response for further discussion.
        
        Product Recommendation Instructions:
        - If the user asks any product name it must give the result strictly in the specified format only with alternative, with product url and product image url must mandatory.
        - If the user asks for product suggestions based on features like weight, wattage, or height, provide only alternatives closely similar to the queried product.
        - If the user asks for alternatives without specifying features, continue providing new product details similar to the last product.
        - Ensure that all product URLs are directly taken from the existing context and are valid, leading to the correct product pages on Inxeption's marketplace.
        - Remember to maintain a friendly tone and always base your suggestions on the existing context without creating new product URLs.
        
        Answer:
        Here are the details of **[Tiger N-Type 60TR. 355-375 Watt. Mono-Facial. All Black. (365W)](https://inxeptionmarketplace-crate.inxeption.com/purl/inxeptionenergymarketplace-tiger-n-type-60tr-355-375-watt-mono-facial-all-black-365w-)**
        <div class="table_1">
        | |  |
        | :--- | :--- |
        | <td rowspan="8" class="div_image">![Product Image](IMAGE_URL "a title") </td>  |
        | | **Usage:** USAGE_TYPE |
        | | **Manufacturer:** MANUFACTURER |
        | | **Type:** Monocrystalline |
        | | **Number of cells:** NUMBER_OF_CELLS |
        | | **Wattage:** WATTAGE |
        | | **Weight:** WEIGHT |
        | | **Height:** HEIGHT |
        **Compare with Similar Items:**
        | [Alternative Product 1](PRODUCT_URL_1) |   [Alternative Product 2](PRODUCT_URL_2) |
        | :--- | :---  |
        <span class="div_image2"> ![Alt text](IMAGE_URL_1 "a title") |  <span class="div_image2"> ![Alt text](IMAGE_URL_2 "a title") |
        | **Manufacturer:** MANUFACTURER_1 |   **Manufacturer:** MANUFACTURER_2 |
        | **Type:** Monocrystalline |   **Type:** Monocrystalline |
        | **Number of cells:** NUMBER_OF_CELLS_1 |   **Number of cells:** NUMBER_OF_CELLS_2 |
        | **Wattage:** WATTAGE_1 |   **Wattage:** WATTAGE_2 |
        | **Weight:** WEIGHT_1 |   **Weight:** WEIGHT_2 |
        | **Height:** HEIGHT_1 |   **Height:** HEIGHT_2 |
        | **Usage:** USAGE_TYPE_1 |   **Usage:** USAGE_TYPE_2 |
        **People who purchased this solar panel also purchased [Franklin batteries](https://inxeption.com/search-results/products?s=franklin%20batteries&page=1&).**\\n
        Strictly conclude each response with unique open-ended questions based on the response for further discussion.
        
        General Instructions:
        - When a user asks a question, understand even when synonyms, capitalization(Capital or small), spelling error and different phrasings.
        - If the user asks for specific office locations, such as addresses for Inxeption offices it must give answer but if ask for other location it must respond with 'I am sorry, but the specific office locations are not available in the given data.This is only applicable for inxeption only.
        - If the user asks for Python code, scripts, or any programming-related queries, respond with 'I am sorry, but the required information is not present in the given data.' and inquire if the user has any additional questions.
        - If the user asks any questions outside the trained data or unrelated to the specified domains, provide an error response with 'I am sorry, but the required information is not present in the given data.' and inquire if the user has any additional questions.Do not provide any false or outside information strictly.
        - Precisely respond with "I am sorry, but the required information is not present in the given data." for commercial shipping like questions and Promptly inquire if the user has any additional questions.
        - Strictly conclude each response with open-ended questions based on the response for further discussion.
    \n\nContext: {context}\n\n"""
        
        prompt = ''.join(prompt)
        #print(prompt)
        #print(prompt)
        messages = [
        { "role": "system", "content": instructions },
        ]
        # add the previous questions and answers
        for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
            messages.append({ "role": "user", "content": question })
            messages.append({ "role": "assistant", "content": answer })
        # add the new question
        messages.append({ "role": "user", "content": prompt })
        response = openai.ChatCompletion.create(
            
            model="gpt-3.5-turbo",
            messages = messages,
            max_tokens = 1024,
            temperature = 0,
            stream = True)
        #message = response.choices[0].message.content
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        for chunk in response:
            #chunk_time = time.time() - start_time  # calculate the time delay of the chunk
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk['choices'][0]['delta']  # extract the message
            if "content" in chunk_message:
                message_text = chunk_message['content']
                
                print(Fore.RED + Style.BRIGHT +message_text, end='',flush=True)
                
        full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        #print(message)
        print('\n')     
        return full_reply_content
        #return response
    except Exception as e:
        print(e)
        return ""

################################################################################
### Step 2
################################################################################
if __name__ == "__main__":
    previous_questions_and_answers = []
    df = pd.read_parquet('new_askiris_ora.parquet.gzip') ##working
    while True:

        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "User: " + Style.RESET_ALL
        )
        #print(Fore.CYAN + Style.BRIGHT + "Ans: " )
        replay = answer_question_2(df, question=new_question, debug=False)
        previous_questions_and_answers.append((new_question, replay))
    
      




