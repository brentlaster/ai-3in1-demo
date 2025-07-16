
## Demo 1

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



   
