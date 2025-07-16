
## Demo 1 - Running models locally

1. Change dir

```
cd ~/ai-3in1-demo
```

2. run ollama

```
ollama run &
```

3. Look up llama3.2  [https://ollama.com](https://ollama.com)

4. pull llama3.2

```
ollama pull llama3.2
```

5. run ollama

```
ollama run llama3.2
```

6. query

```
What is the weather in Paris?
```

7. curl for api access

```
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What causes weather changes?",
  "stream": false
}'
```

## Demo 2 - Running an agent

1. View code for [agent.py](./agent.py)

2. Run the agent

```
python agent.py
```

3. Put in a location and observe tooling calls


## Demo 3 - Vectors

1. View the PDF file  [data/offices.pdf](data/offices.pdf)

2. View the indexing file [tools/index_pdf.py](tools/index_pdf.py)

3. Run the indexer

```
python index_pdf.py
```

4. View local chromadb

5. View the search file [tools/search.py](tools/search.py)

6. Run the search

```
python search.py
```

7. Prompt about offices

```
Tell me about HQ
Tell me about the Southern office
```

## Demo 4 - Agent with RAG

1. Look at rag agent code [rag_agent.py](rag_agent.py]

2. Run rag_agent.py

```
python rag_agent.py
```

3. Prompt about HQ, Southern office etc.

```
Tell me about HQ
```

4. View rag_agent2 code [rag_agent2.py](rag_agent2.py)

5. Run rag_agent2.py

```
python rag_agent2.py
```

6. Prompt about HQ, etc.
   





   
