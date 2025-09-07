# LangChain – Complete Guide

LangChain is a powerful framework for building applications with **Large Language Models (LLMs)**. Instead of just calling an LLM, LangChain provides **tools, abstractions, and integrations** to connect models with data, memory, tools, and workflows.

---

## 🌐 Why LangChain?
- Standardizes interaction with LLMs across providers.
- Allows chaining prompts, tools, and logic.
- Provides memory to make AI conversational.
- Integrates with databases, APIs, vector stores, and external tools.
- Foundation for building **agents and real-world AI products**.

---

## 📍 Roadmap to Learn LangChain
1. **Basics** – LLMs, Chat Models, Prompt Templates, Chains.
2. **Memory** – Short-term & long-term memory for conversational agents.
3. **Tools & Agents** – Adding external functionalities (search, calculator, APIs).
4. **RAG (Retrieval Augmented Generation)** – Connecting LLMs with knowledge bases.
5. **Evaluation & Debugging** – Understanding outputs, improving prompts.
6. **Integration** – With vector databases, APIs, frontends.

---

## 📘 Important Concepts

### 1. LLMs & Chat Models
- **LLM**: Large text-in, text-out models (e.g., GPT, Gemini).
- **Chat Models**: Conversation-oriented models (history, roles).

### 2. Prompt Templates
- Standardize prompts.
- Example: `PromptTemplate.from_template("Translate {text} to French")`.

### 3. Chains
- Sequence of calls (LLM → Tool → Output).
- Types: `SimpleChain`, `SequentialChain`, `RouterChain`.

### 4. Memory
- Lets models “remember” past conversations.
- Types: `ConversationBufferMemory`, `ConversationSummaryMemory`, `VectorStoreRetrieverMemory`.

### 5. Tools & Agents
- **Tools**: External actions (API calls, DB queries).
- **Agents**: Decide which tool to use at runtime.
- Types: `Zero-shot-react-description`, `Conversational Agent`, `Self-ask with Search`.

### 6. RAG (Retrieval Augmented Generation)
- Combine LLMs with **retrievers** (vector DBs like FAISS, Pinecone, Chroma).
- Steps: Document Loader → Text Splitter → Embeddings → Retriever → Chain.

### 7. Evaluation
- LangSmith + built-in evaluation for output quality, token cost, etc.

---

## 🤖 LLM Models You Can Work With
- **OpenAI**: GPT-3.5, GPT-4
- **Anthropic**: Claude
- **Google**: Gemini (Pro, Ultra)
- **Meta**: LLaMA 2
- **Mistral**: Mistral 7B, Mixtral
- **Cohere**: Command R+
- **Local Models**: LLaMA.cpp, GPT4All, HuggingFace models

💡 Pro Tip: Always wrap your LLMs with LangChain’s `ChatOpenAI`, `ChatAnthropic`, etc. → this makes swapping models easy.

---

## 🛠️ How to Use LangChain Practically

### Example 1 – Simple Prompt Chain
```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate.from_template("Translate this English text to French: {text}")
chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("Hello, how are you?"))
```

### Example 2 – RAG with Vector DB
```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS

# Load retriever (vector DB)
retriever = FAISS.load_local("./my_index", embeddings)

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=retriever,
    chain_type="stuff"
)

print(qa.run("What did the document say about AI?"))
```

### Example 3 – Agent with Tools
```python
from langchain.agents import initialize_agent, load_tools
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("Who is the CEO of Google and what is 25*4?")
```

---

## 🌍 Real-World Projects with LangChain
- **AI Customer Support Bot** (RAG + Agents + Memory).
- **Research Assistant** (multi-document summarizer).
- **Personal Productivity Agent** (calendar + notes + focus app integration).
- **AI Tutor** (interactive learning assistant).
- **Business Automation Agent** (fetch reports, summarize, email automation).

---

## ⚡ Tips & Tricks
- Start with **simple chains** before jumping into agents.
- Use **LangSmith** for debugging & evaluation.
- Optimize token usage (short prompts, embeddings, RAG).
- Keep workflows modular (separate chains → combine later).
- Experiment with **different LLMs** for cost-performance balance.

💡 **Pro Tip:** Learn by replicating **real-world SaaS features** (chatbot, summarizer, Q&A system). This builds both skills and portfolio.

---

## 📚 Useful Resources
- [LangChain Docs](https://python.langchain.com/)
- [LangChain YouTube](https://www.youtube.com/@LangChain)
- [LangChain Hub](https://smith.langchain.com/hub)
- [Full-Stack AI Tutorials](https://www.youtube.com/@fullstackai)
- [RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)

---

## ✅ Summary
- **Core concepts**: LLMs, Prompts, Chains, Memory, Agents, Tools, RAG.
- **Roadmap**: Basics → Memory → Tools → RAG → Evaluation → Deployment.
- **Practical usage**: Build chains, agents, retrievers, and deploy real apps.
- **Goal**: Use LangChain not just for experiments but for **production-ready AI systems**.
