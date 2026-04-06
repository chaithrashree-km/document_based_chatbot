from groq import Groq
from app.Config import Config
import logging

class Response:

 config = Config()
 client = Groq(api_key=config.GROQ_API_KEY)

 logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

 def generate_response(self, question: str, context: str):

    response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": f"""You are a precise document assistant. Use the context below to answer 
                 questions. Follow the decision logic exactly:

                 ┌─ Is it a coding/programming question?
                 │   YES → "I'm a document assistant and not designed to answer coding 
                 │           or technical programming questions."    
                 │
                 └─ Is it a factual question?
                 Search context for the answer and related synonyms.
                 FOUND → Answer fully with all specifics. 
                         Do NOT embed citations mid-sentence.
                         At the very end of the answer, add ONE consolidated citation block:
                            - Same document, one page   → (Source: <filename>, Page: <page>)
                            - Same document, many pages → (Source: <filename>, Page: <p1> & <p2> & ...)
                            - Multiple documents        → one line per document, same format above.
                         Do not include duplicate citation.
                 PARTIAL → Share what is available, note what is missing.
                 NOT FOUND → "The documents does not have a specific answer to your question."

                 Context: {context}
                """
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    logging.info("RESPONSE USAGE: %s", response.usage)
    return response.choices[0].message.content
 
 def detect_intent(self, question: str):
    response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """ You are an intent classification engine. Given a natural language user query, 
                          identify what the user is fundamentally trying to accomplish — explicitly or implicitly.
                          Derive the intent from the semantics, tone, and context of the message.
                          The final intent must be 2-3 word for chat title relevant to the user query.
                """
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content.strip()
 
 def generate_summary_response(self, question: str, context: str):
    response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Document Summarizer assistant. Use the context below to generate a summary. "
                    "Do not answer outside of the context.\n\n"
                    "Rules:\n"
                    "1. Only summarize the exact document the query mentions.\n"
                    "2. Summarize all major points.\n"
                    "3. Add the source citation ONLY at the very end of the summary.\n"
                    "4. Do NOT add source/page citations mid-paragraph.\n\n"
                    "Context:\n"
                    f"{context}"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    logging.info("RESPONSE USAGE: %s", response.usage)
    return response.choices[0].message.content


 def generate_inventory_response(self, question: str, context: str):
    response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Document Listing assistant. Use the context below to list available documents. "
                    "Do not answer outside of the context.\n\n"
                    "Rules:\n"
                    "- Start your response with exactly: 'Here is a list of the documents I have with a one-sentence description each:'\n"
                    "- List each document on a new line as: <number>. <filename>: <one sentence, max 20 words>.\n"
                    "- Leave a blank line between each document.\n"
                    "- Do not continue a new document on the same line as the previous one.\n"
                    "- Do not use markdown, bullet points, headers, or bold text.\n"
                    "- Do not wrap filenames or any text in ** or other markdown symbols.\n"
                    "- Plain text only.\n\n"
                    "Context:\n"
                    f"{context}"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    logging.info("RESPONSE USAGE: %s", response.usage)
    return response.choices[0].message.content

 