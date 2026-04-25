# 🧠 SriLab.AI India, HR Policy - Personal RAG Assistant

Welcome to the Master RAG Control Tower! This is a complete, user-friendly desktop application that allows you to upload your own private documents (PDFs, Word files, Text files) and chat directly with them using Google's powerful Gemini AI.

---

## 🧠 Part 1: Understanding RAG (For Beginners)

### What is RAG?
RAG stands for **Retrieval-Augmented Generation**. It is a modern technique used to make Artificial Intelligence smarter, more accurate, and aware of *your* specific private data.

### The Layman's Example: The "Open-Book Exam"
Imagine a highly intelligent student (the AI) taking a test.

* **Standard AI (No RAG):** The student takes a *closed-book* exam. If you ask them a general question ("What is the capital of France?"), they know it from memory. But if you ask them a highly specific question ("What is the vacation policy for *my* specific company?"), they don't know. To avoid failing, they might guess or make something up. In AI, this making-things-up is called a **Hallucination**.
* **AI with RAG:** The student is taking an *open-book* exam. Before answering your specific question, the student first goes to the library (**Retrieval**), finds the exact company handbook you gave them, reads the relevant highlighted paragraph (**Augmented**), and then writes down the perfect answer (**Generation**).

RAG simply means: **Retrieve** the right document, **Augment** (feed) it to the AI, and let the AI **Generate** the answer based *only* on that document.

### The 4 Steps of RAG:
1. **Ingestion:** You upload a file.
2. **Chunking:** The system cuts the file into small paragraphs so the AI doesn't get overwhelmed reading a 1,000-page book all at once.
3. **Embedding:** The system translates these human words into "math vectors" (numbers) so the computer can understand the meaning of the text.
4. **Retrieval & Chat:** When you ask a question, the system finds the math vectors that best match your question, grabs the text, and gives it to Gemini to read and answer.

---

## 🗺️ Part 2: Architecture & Workflow

The system is divided into two main phases. Here is exactly how data flows through the application:

### PHASE 1: KNOWLEDGE INGESTION (Sidebar Architecture)
This phase runs when you upload a document and click **"Update Knowledge Base"**.

```
┌─────────────────────────────┐
│ 1. User Uploads Document    │ (PDF, TXT, DOCX)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 2. Detect File Extension    │
└──────────────┬──────────────┘
               ├──────────────────────────────┐
               ▼                              ▼
┌─────────────────────────────┐  ┌────────────────────────────┐
│ 3. Execute Document Loader  │  │ Abort & Display Error      │◄── (If unsupported
│    (Uses UTF-8 for TXT)     │  └────────────────────────────┘     or empty)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 4. Recursive Text Splitter  │◄── Uses Sidebar Inputs:
└──────────────┬──────────────┘    [Chunk Size] & [Chunk Overlap]
               ▼
┌─────────────────────────────┐
│ 5. Chunk Sanitizer Filter   │◄── Removes empty lines/borders
└──────────────┬──────────────┘    to prevent database crashes
               ├──────────────────────────────┐
               ▼                              ▼
┌─────────────────────────────┐  ┌────────────────────────────┐
│ 6. Generate Embeddings      │──► Update "Processing Details"│
│    (text-embedding-004)     │  │ UI Table in Sidebar        │
└──────────────┬──────────────┘  └────────────────────────────┘
               ▼
╔═════════════════════════════╗
║ 7. Local ChromaDB Vector    ║◄── Saved to ./chroma_db_store
║    Storage                  ║
╚═════════════════════════════╝
```

### PHASE 2: RAG QUERY & RESPONSE (Main Chat Architecture)
This phase runs when you type a question into the **Chat Input** box.

```
┌─────────────────────────────┐
│ 1. User Types Prompt        │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 2. Embed User Question      │◄── Uses same Embedding Model
└──────────────┬──────────────┘    to create a search vector
               ▼
╔═════════════════════════════╗
║ 3. Query Local ChromaDB     ║──► Returns Top 3 Most Relevant
╚═════════════════════════════╝    Document Chunks (Context)
               │
               ▼
┌─────────────────────────────┐
│ 4. Assemble Final Prompt    │◄── Injects Context & Question
└──────────────┬──────────────┘    into strict System Prompt
               ▼
┌─────────────────────────────┐
│ 5. Execute LLM Generation   │◄── Uses Sidebar Input:
│    (e.g., gemini-1.5-flash) │    [Chat Model Selection]
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 6. Calculate Exact Tokens   │◄── Bypasses LangChain; uses Native
│    (Native Client count)    │    Google GenAI Client for accuracy
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 7. Render Final UI Output   │◄── Displays Answer + 📊 Token Stats
└─────────────────────────────┘
```

---

## 🛠️ Part 3: Setup and Execution Commands

Follow these exact steps to run the application on your machine.

### 1. Prerequisites
Ensure you have **Python 3.9 or higher** installed on your computer.

### 2. Create the Requirements File
Create a file named `requirements.txt` in your project folder and paste the following exact text into it:

```text
# Modern Unified Google SDK
google-genai>=1.0.0
langchain-google-genai>=2.0.0

# Core RAG & Vector DB
langchain>=0.3.0
langchain-chroma>=0.1.0
langchain-text-splitters>=0.0.1
langchain-classic>=0.0.1
chromadb>=0.5.0

# Document Parsing & UI
pypdf>=4.0.0
docx2txt>=0.8
streamlit>=1.30.0
pandas>=2.2.0
```

### 3. Install Dependencies
Open your terminal (or Command Prompt), navigate to your project folder, and run:

```bash
pip install -r requirements.txt
```

### 4. Run the Application
Once the installation is complete, start the server by running:

```bash
streamlit run ragv2.py
```

A browser window will automatically open with your Master RAG Control Tower. If it asks for an email, you can leave it blank and press Enter.

---

## 🐍 Part 4: Python Script Breakdown (Block by Block)

Here is a detailed, plain-English explanation of what every section of your `ragv2.py` script does.

### Block 1: The Imports (Getting our Tools)

```python
import os
import tempfile
import logging
import streamlit as st
import pandas as pd
from io import StringIO
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
```

**What it does:** Think of this as opening your toolbox before building a house.

- `streamlit` builds the beautiful web page (buttons, sidebars).
- `pandas` manages the data tables (like the chunk inspector).
- `genai` connects us to Google's brain (using the new unified SDK).
- `langchain` is the glue that connects our documents, the database, and the AI together so they can talk to each other.
- `Chroma` is our local database where we store the document chunks.

---

### Block 2: Dynamic Model Fetching

```python
def get_available_models(api_key):
    try:
        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
        
        chat_models = []
        embed_models = []
        
        for m in models:
            actions_str = ""
            if hasattr(m, 'supported_actions') and m.supported_actions:
                actions_str = str(m.supported_actions).lower()
            elif hasattr(m, 'supported_generation_methods') and m.supported_generation_methods:
                actions_str = str(m.supported_generation_methods).lower()
                
            if 'generate' in actions_str: chat_models.append(m.name)
            if 'embed' in actions_str: embed_models.append(m.name)
                
        return chat_models, embed_models
    except Exception as e:
        return [], []
```

**What it does:** Google frequently updates their AI models. Instead of hard-coding a model name that might break next year, this block logs into your Google account behind the scenes, asks Google "What AI models are available today?", and populates your sidebar dropdowns with the freshest list safely. The `hasattr` checks prevent the script from crashing if Google changes variable names.

---

### Block 3: The Vector Store (The Librarian)

```python
@st.cache_resource
def get_vector_store(api_key, embed_model):
    os.environ["GOOGLE_API_KEY"] = api_key
    if not embed_model.startswith("models/"):
        embed_model = f"models/{embed_model}"
        
    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model)
    return Chroma(persist_directory="./chroma_db_store", embedding_function=embeddings)
```

**What it does:** This creates our local database (named `chroma_db_store`). It uses the "Embedding Model" (like `text-embedding-004`) to translate human words into numbers and stores them on your hard drive. `@st.cache_resource` is a trick that tells the app to remember this database so it doesn't have to rebuild it every time you click a button.

---

### Block 4: Processing the Uploaded File (The Meat Grinder)

```python
def process_uploaded_file(uploaded_file, vector_store, c_size, c_overlap):
    # 1. Load the file
    ext = uploaded_file.name.split('.')[-1].lower()
    # ... temp file creation ...
    if ext == 'pdf': loader = PyPDFLoader(temp_file_path)
    elif ext == 'txt': loader = TextLoader(temp_file_path, encoding='utf-8')
    elif ext == 'docx': loader = Docx2txtLoader(temp_file_path)
    
    raw_docs = loader.load()
    
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    raw_chunks = text_splitter.split_documents(raw_docs)
    
    # 3. Sanitize and Filter
    chunked_docs = []
    for chunk in raw_chunks:
        clean_text = chunk.page_content.replace('═', '').strip()
        if len(clean_text) > 5:
            chunked_docs.append(chunk)
            
    # 4. Save to ChromaDB
    vector_store.add_documents(chunked_docs)
```

**What it does:** This is the most important part of the ingestion phase.

- It looks at your file (e.g., PDF or UTF-8 encoded text) and reads the text.
- It takes a massive wall of text and cuts it into chunks based on your sidebar inputs. Why overlap? If a sentence gets cut in half at the end of Chunk A, the "Overlap" ensures the beginning of Chunk B repeats that sentence so context is never lost.
- **Sanitization:** It deletes empty chunks (like pure lines of symbols `════` or extra spaces) to prevent the database from crashing with "index out of range" errors.
- Finally, it saves these clean chunks into your Chroma database.

---

### Block 5: The Sidebar UI (The Control Panel)

```python
with st.sidebar:
    st.header("⚙️ System Settings")
    api_key = st.text_input("Gemini API Key:", type="password")
    
    # Dropdowns populated by the Dynamic Model Fetcher
    chat_selection = st.selectbox("Chat Model", c_models, index=c_index)
    embed_selection = st.selectbox("Embedding Model", e_models, index=e_index)
    
    # Numeric sliders for chunking
    c_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=1000, step=100)
    c_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200, step=50)
    
    uploaded_file = st.file_uploader("Upload Knowledge", type=['pdf', 'txt', 'docx'])
    
    # UI Table for Log Inspection
    st.dataframe(st.session_state.last_chunk_log, hide_index=True)
```

**What it does:** This builds the left-hand panel of your web app. It provides user-friendly dropdowns, sliders, and buttons so you don't have to edit code to change how the system works. It also includes the "Processing Details" data table so you can visually inspect the chunks the computer created!

---

### Block 6: The Main Chat Interface (The Conversation)

```python
if prompt := st.chat_input("Ask about your uploaded data..."):
    # 1. Retrieve documents from the database
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # 2. Create a strict prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Answer strictly using the context below. If missing, say you don't know.\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])
    
    # 3. Connect to Chat Model and Ask Gemini
    llm = ChatGoogleGenerativeAI(model=c_model_formatted, temperature=0.3)
    qa_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    response = rag_chain.invoke({"input": prompt})
```

**What it does:** When you type a question and hit enter:

1. The system searches your Chroma database and grabs the top 3 (`k=3`) most relevant chunks of text based on your prompt.
2. It builds a secret instruction for the AI (the `prompt_template`) that essentially says: "Read these 3 chunks. Now answer the human's question using ONLY these chunks. Do not make anything up."
3. It sends this package to the selected Gemini model and prints the generated answer on your screen.

---

### Block 7: Token Tracking (The Auditor)

```python
# Bypassing LangChain to get the exact counts directly from the Google API
try:
    native_client = genai.Client(api_key=api_key)
    full_input_context = prompt + "\n" + "\n".join([doc.page_content for doc in response["context"]])
    
    in_tokens = native_client.models.count_tokens(
        model=c_model_formatted, contents=full_input_context
    ).total_tokens
    
    out_tokens = native_client.models.count_tokens(
        model=c_model_formatted, contents=response["answer"]
    ).total_tokens
    
    token_info = {"in": in_tokens, "out": out_tokens}
except Exception as e:
    token_info = {"in": "Error", "out": "Error"}

st.caption(f"📊 Token usage: Input {token_info['in']} | Output {token_info['out']}")
```

**What it does:** This block is your personal AI usage meter — like the electricity meter on the side of your house, but for AI brain-power.

#### 🪙 What Exactly is a "Token"?

Forget bytes and bits. Think of tokens like **poker chips at a casino**. Every word you send to the AI, and every word it sends back, costs a certain number of chips. Google counts these chips and charges you (or deducts from your free quota) based on how many were used.

Here is a rough feel for the size of a token in plain English:

| What you write | Approx. tokens used |
|---|---|
| The word `"cat"` | 1 token |
| The sentence `"What is the refund policy?"` | ~6 tokens |
| A full paragraph (100 words) | ~75 tokens |
| A 10-page PDF document | ~3,000–5,000 tokens |
| An entire novel (80,000 words) | ~60,000 tokens |

> 💡 **Quick rule of thumb:** 1 token ≈ ¾ of a word. Or flip it: every 4 tokens ≈ roughly 3 words of English text.

---

#### 📬 Input Tokens vs. Output Tokens — The Postage Analogy

Imagine you are sending a physical letter to a very smart friend who lives overseas and charges for their time.

- **Input Tokens = The weight of the letter you send.**
  Everything you stuff into the envelope counts: your question, the 3 document chunks retrieved from the database, and the secret system instructions ("Answer only from the context..."). The heavier the envelope, the more it costs to send.

- **Output Tokens = The length of the reply letter they write back.**
  If your friend writes a short reply ("The refund window is 30 days."), that is cheap. If they write a 5-page essay, that costs more.

In your RAG app specifically:

- **Input** = Your typed question **+** the 3 retrieved document chunks **+** the system prompt instructions
- **Output** = The final answer Gemini writes back to you

---

#### 🔍 Why Does This Block Bypass LangChain?

LangChain is a brilliant middleman — it connects your documents, database, and AI together effortlessly. But middlemen sometimes lose the receipt. LangChain wraps the Google API in so many helpful layers that the exact token counts get obscured or simply aren't surfaced.

Think of it like ordering food through a delivery app: the restaurant knows exactly how many ingredients they used, but the app's summary receipt might just say "1 x Meal Deal." This block throws away the delivery app's vague receipt and calls the restaurant (Google) directly to ask: *"Exactly how many ingredients did you use?"*

That is what `native_client.models.count_tokens(...)` does — it dials Google's API directly and gets the mathematically precise count, not an estimate.

---

#### 💰 Why Should You Care About Token Counts?

1. **Cost control:** Google's free tier has a monthly token budget. If you upload a massive 500-page legal document and ask 50 questions about it, you might burn through your quota without realising it. This meter tells you exactly how fast you're spending.

2. **Context window limits:** Every AI model has a maximum number of tokens it can hold in its "head" at once (its *context window*). If your input tokens exceed this limit, the AI literally cannot read the whole thing — it's like handing someone a 10-page document and they can only read the first 3 pages. Watching your input token count helps you stay safely within bounds.

3. **Debugging quality:** If you ask a question and get a vague answer, checking the input token count tells you whether the retrieved chunks were actually substantial (high token count = lots of context was found) or nearly empty (low token count = the database had almost nothing relevant to pull in).

The final line, `st.caption(f"📊 Token usage: Input {token_info['in']} | Output {token_info['out']}")`, prints this live meter neatly beneath every single chat reply, so you always know exactly what each conversation turn cost you.

---

## 🧪 Part 5: Sample Knowledge Base — SriLab.AI HR Policy

To help you get started immediately without hunting for test documents, this project ships with a ready-made sample knowledge base: the **SriLab.AI HR Policy Manual & Compensation Guide** (`SriLab_AI_HR_Policies-v3.txt`).

> SriLab.AI is a fictional Indian AI startup used purely for demonstration purposes. The document is realistic enough to exercise every part of the RAG pipeline — tables, legal clauses, numbers, gender-specific policies, and compensation math — making it an ideal first upload.

### 📄 What's Inside the Document?

| Section | What It Covers |
|---|---|
| **Welcome Messages** | Notes from the CEO (Sri) and HR Head (Heer) |
| **General Workplace Policies** | Working hours, leave types (EL / CL / SL), public holidays |
| **Gender-Specific Labour Laws** | Maternity Act 2017, POSH Act 2013, Paternity Leave, night-shift transport |
| **Common Statutory Compliances** | Equal Remuneration Act, EPF Act, Payment of Gratuity Act |
| **Compensation & CTC Breakdown** | Step-by-step salary formulas, component percentages, worked example at ₹12 LPA |

### 🗂️ How to Upload It

1. Launch the app with `streamlit run ragv2.py`
2. Enter your Gemini API Key in the sidebar
3. Click **"Browse files"** under *Upload Knowledge* and select `SriLab_AI_HR_Policies-v3.txt`
4. Click **"Update Knowledge Base"** and wait for the Processing Details table to populate
5. Start asking questions in the chat box below

---

### 💬 Sample Prompt Questions (Copy & Paste Ready)

Use these prompts to test the system right after uploading the HR policy file. They are ordered from simple lookups to complex multi-step reasoning.

#### 🟢 Beginner — Simple Fact Retrieval

These questions have a single, direct answer inside the document. Gemini should answer instantly with high confidence.

```
How many days of Earned Leave does SriLab.AI provide per year?
```
> *Expected answer: 15 days per annum.*

```
What are the standard working hours at SriLab.AI?
```
> *Expected answer: 9:00 AM to 6:00 PM, Monday to Friday (40 hours per week).*

```
Who is the HR Head at SriLab.AI?
```
> *Expected answer: Heer.*

```
How many weeks of paid maternity leave does SriLab.AI offer for a first child?
```
> *Expected answer: 26 weeks, under the Maternity Benefit (Amendment) Act, 2017.*

---

#### 🟡 Intermediate — Policy Reasoning

These require the AI to read a clause and explain it in context, not just copy-paste a number.

```
What happens if a female employee at SriLab.AI needs to work after 8 PM? What support does the company provide?
```
> *Expected: Company-arranged secure door-to-door transportation between 8:00 PM and 6:00 AM.*

```
Does SriLab.AI have a harassment policy for male employees, or is it only for women?
```
> *Expected: SriLab.AI enforces a gender-neutral anti-harassment policy; male complaints are handled by the HR Disciplinary Committee with equal severity.*

```
After how many years of service does an employee become eligible for Gratuity at SriLab.AI?
```
> *Expected: After 5 years of continuous service, under the Payment of Gratuity Act, 1972.*

```
What is the Internal Complaints Committee (ICC) and who leads it?
```
> *Expected: A body constituted under the POSH Act to resolve harassment grievances, headed by a senior female employee.*

---

#### 🔴 Advanced — Compensation Math & Multi-Step Reasoning

These test whether the RAG system correctly retrieves the salary formula tables and whether the AI can apply multi-step logic.

```
If my Total Fixed CTC is ₹12,00,000 per annum, what will my monthly Basic Salary be?
```
> *Expected: ₹46,122 per month (50% of Gross Salary, where Gross = CTC ÷ 1.0841).*

```
Explain the difference between Gross Salary and Total CTC at SriLab.AI. What two components make up the gap?
```
> *Expected: Gross Salary is the take-home-related component. Total CTC = Gross Salary + Employer PF + Gratuity. The gap is the statutory retiral benefits.*

```
What percentage of my CTC goes toward statutory retiral benefits (EPF + Gratuity) according to SriLab.AI's compensation policy?
```
> *Expected: ~7.75% of Total CTC (EPF ~5.53% + Gratuity ~2.22%).*

```
A male employee's wife just had a baby. How many days of paid leave is he entitled to, and under which policy?
```
> *Expected: 14 calendar days (2 weeks) of paid paternity leave under SriLab.AI's company policy aligned with progressive standards.*

---

#### 🔵 Hallucination Trap — Out-of-Scope Questions

These questions have **no answer** in the document. A well-configured RAG system should politely say *"I don't know based on the provided documents"* rather than making something up. Use these to verify your system prompt is working correctly.

```
What is the stock price of SriLab.AI?
```

```
Does SriLab.AI offer Employee Stock Options (ESOPs)?
```

```
What is the appraisal cycle frequency at SriLab.AI?
```

```
Who is the CTO of SriLab.AI?
```

> ✅ **If the AI says "I don't have information about this in the provided documents"** — your RAG system is working perfectly. It is grounded and hallucination-resistant.
>
> ❌ **If the AI confidently makes up an answer** — revisit your system prompt in Block 6 to ensure it strictly instructs the model to answer only from the retrieved context.

---

### 📊 Expected Token Usage (Rough Guide)

| Prompt Type | Approx. Input Tokens | Approx. Output Tokens |
|---|---|---|
| Simple fact retrieval (Beginner) | 400 – 700 | 30 – 80 |
| Policy reasoning (Intermediate) | 600 – 900 | 80 – 200 |
| Compensation math (Advanced) | 700 – 1,100 | 150 – 350 |
| Out-of-scope / Hallucination trap | 300 – 500 | 20 – 60 |

> These are estimates based on `k=3` chunks retrieved and default chunk size of 1,000. Your actual numbers will vary slightly based on sidebar settings.

## Application Runtime screenshots
####
### <img width="476" height="578" alt="1" src="https://github.com/user-attachments/assets/2ac30d8a-8b47-44b6-a48d-b7890280efec" />
####
### <img width="470" height="742" alt="2" src="https://github.com/user-attachments/assets/8620a20e-0e23-443f-9771-9b96bf972a6d" />
####
### <img width="1182" height="863" alt="3" src="https://github.com/user-attachments/assets/6b8c6f87-19f3-4e53-8d1a-fe3c147c8c20" />
