# AI 3-in-1: Agents, RAG and Local Models
## Building out an AI agent that uses RAG and runs locally
## Session labs 
## Revision 3.8 - 07/06/25

**Follow the startup instructions in the README.md file IF NOT ALREADY DONE!**

**NOTE: To copy and paste in the codespace, you may need to use keyboard commands - CTRL-C and CTRL-V. Chrome may work best for this.**

**Lab 1 - Using Ollama to run models locally**

**Purpose: In this lab, we’ll start getting familiar with Ollama, a way to run models locally.**

1. We already have a script that can download and start Ollama and fetch some models we'll need in later labs. Take a look at the commands being done in the *../scripts/startOllama.sh* file. 
```
cat scripts/startOllama.sh
```

2. Go ahead and run the script to get Ollama and start it running.
```
./scripts/startOllama.sh &
```

The '&' at the end will causes the script to run in the background. You will see a set of startup messages. After those, you can just hit *Enter* to get back to a prompt in the terminal.

![starting ollama](./images/31ai6.png?raw=true "starting ollama")

3. Now let's find a model to use. Go to https://ollama.com and in the *Search models* box at the top, enter *llama*. In the list that pops up, choose the entry for "llama3.2".

![searching for llama](./images/31ai7.png?raw=true "searching for llama")

4. This will put you on the specific page about that model. Scroll down and scan the various information available about this model.
![reading about llama3.2](./images/31ai8.png?raw=true "reading about llama3.2")

5. Switch back to a terminal in your codespace. While it's not necessary to do as a separate step, first pull the model down with ollama. (This will take a few minutes.)

```
ollama pull llama3.2
```
![pulling the model](./images/31ai9.png?raw=true "pulling the model")

6. Once the model is downloaded, run it with the command below.
```
ollama run llama3.2
```

7. Now you can query the model by inputting text at the *>>>Send a message (/? for help)* prompt.  Let's ask it about what the weather is in Paris. What you'll see is it telling you that it doesn't have access to current weather data and suggesting some ways to gather it yourself.

```
What's the current weather in Paris?
```

![answer to weather prompt and response](./images/31ai10.png?raw=true "answer to weather prompt and response")

8. Now, let's try a call with the API. You can stop the current run with a Ctrl-D or switch to another terminal. Then put in the command below (or whatever simple prompt you want). 
```
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What causes weather changes?",
  "stream": false
}'
```

9. This will take a minute or so to run. You should see a single response object returned with lots of data. But you can make out the text answer if you look for it. You can try out some other prompts/queries if you want.

![query response](./images/31ai11.png?raw=true "Query response")

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 2 - Creating a simple agent**

**Purpose: In this lab, we’ll learn about the basics of agents and see how tools are called. We'll also see how Chain of Thought prompting works with LLMs and how we can have ReAct agents reason and act.**

1. In our repository, we have a set of Python programs that we'll be building out to work with concepts in the labs. These are mostly in the *agents* subdirectory. Go to the *TERMINAL* tab in the bottom part of your codespace and change into that directory.
```
cd agents
```

2. For this lab, we have the outline of an agent in a file called *agent.py* in that directory. You can take a look at the code either by clicking on [**agent.py**](./agent.py) or by entering the command below in the codespace's terminal.
   
```
code agent.py
```
![starting agent code](./images/31ai12.png?raw=true "Starting agent code")

3. As you can see, this outlines the steps the agent will go through without all the code. When you are done looking at it, close the file by clicking on the "X" in the tab at the top of the file.

4. Now, let's fill in the code. To keep things simple and avoid formatting/typing frustration, we already have the code in another file that we can merge into this one. Run the command below in the terminal.

```
code -d ../extra/lab2-code.txt agent.py
```

5. Once you have run the command, you'll have a side-by-side in your editor of the completed code and the agent1.py file.
  You can merge each section of code into the agent1.py file by hovering over the middle bar and clicking on the arrows pointing right. Go through each section, look at the code, and then click to merge the changes in, one at a time.

![Side-by-side merge](./images/31ai13.png?raw=true "Side-by-side merge") 

6. When you have finished merging all the sections in, the files should show no differences. Save the changes simply by clicking on the "X" in the tab name.

![Merge complete](./images/31ai14.png?raw=true "Merge complete") 

7. Now you can run your agent with the following command:

```
python agent1.py
```

![Running the agent](./images/31ai15.png?raw=true "Running the agent")

8. The agent will start running and will prompt for a location (or "exit" to finish). At the prompt, you can type in a location like "Paris, France" or "London" or "Raleigh" and hit *Enter*. After that you'll be able to see the Thought -> Action -> Observation loop in practice as each one is listed out. You'll also see the arguments being passed to the tools as they are called. Finally you should see a human-friendly message from the AI summarizing the weather forecast.

![Agent run](./images/31ai16.png?raw=true "Agent run") 

9. You can then input another location and run the agent again or exit. Note that if you get a timeout error, the API may be limiting the number of accesses in a short period of time. You can usually just try again and it will work.

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 3 - Exploring MCP**

**Purpose: In this lab, we’ll see how MCP can be used to standardize an agent's interaction with tools.**

1. We have partial implementations of an MCP server and an agent that uses a MCP client to connect to tools on the server. So that you can get acquainted with the main parts of each, we'll build them out as we did the agent in the first lab - by viewing differences and merging. Let's start with the server. Run the command below to see the differences.

```
code -d ../extra/lab3-server.txt mcp_server.py
```
</br></br>
![MCP server code](./images/31ai17.png?raw=true "MCP server code") 

2. As you look at the differences, note that we are using FastMCP to more easily set up a server, with its @mcp.tool decorators to designate our functions as MCP tools. Also, we run this using the *streamable-http* transport protocol. Review each difference to see what is being done, then use the arrows to merge. When finished, click the "x"" in the tab at the top to close and save the files.

3. Now that we've built out the server code, run it using the command below. You should see some startup messages similar to the ones in the screenshot.

```
python mcp_server.py
```
</br></br>
![MCP server start](./images/31ai18.png?raw=true "MCP server start") 

4. We also have a small tool that can call the MCP discover method to find the list of tools from our server. This is just for demo purposes.  You can take a look at the code either by clicking on [**tools/discover_tools.py**](./tools/discover_tools.py) or by entering the first command below in the codespace's terminal. The actual code here is minimal. It connects to our server and invokes the list_tools method. Run it with the second command below and you should see the list of tools like in the screenshot.

```
code tools/discover_tools.py
python tools/discover_tools.py
```

![Discovering tools](./images/31ai19.png?raw=true "Discovering tools") 
   
5. Now, let's turn our attention to the agent that will use the MCP server through an MCP client interface. First, since the terminal is tied up with the running server, we need to have a second terminal to use to work with the client. So that we can see the server responses, let's just open another terminal side-by-side with this one. To do that, right-click in the current terminal and select *Split Terminal* from the pop-up context menu.

![Opening a second terminal](./images/31ai20.png?raw=true "Opening a second terminal") 

6. In the second terminal, run a diff command so we can build out the new agent.

```
code -d ../extra/lab3-code.txt mcp_agent.py
```

7. Review and merge the changes as before. What we're highlighting in this step are the *System Prompt* that drives the LLM used by the agent, the connection with the MCP client at the /mcp/ endpoint, and the mpc calls to the tools on the server. When finished, close the tab to save the changes as before.

![Agent using MCP client code](./images/31ai21.png?raw=true "Agent using MCP client code") 
   
7. After you've made and saved the changes, you can run the client in the terminal with the command below.

```
python mcp_agent.py
```

8. The agent should start up, and wait for you to prompt it about weather in a location. You'll be able to see similar TAO output. And you'll also be able to see the server INFO messages in the other terminal as the MCP connections and events happen. A suggested prompt is below.

```
What is the weather in New York?
```

![Agent using MCP client running](./images/31ai22.png?raw=true "Agent using MCP client running") 

9. When you're done, you can use 'exit' to stop the client and CTRL-C to stop the server. 

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 4 - Working with Vector Databases**

**Purpose: In this lab, we’ll learn about how to use vector databases for storing supporting data and doing similarity searches.**

1. In our repository, we have a simple program built around a popular vector database called Chroma. The file name is vectordb.py. Open the file either by clicking on [**genai/vectordb.py**](./genai/vectordb.py) or by entering the command below in the codespace's terminal.

```
code vectordb.py
```

2. For purposes of not having to load a lot of data and documents, we've *seeded* the same data strings in the file that we're loosely referring to as *documents*. These can be seen in the *datadocs* section of the file.
![data docs](./images/gaidd47.png?raw=true "Data docs")

3. Likewise, we've added the metadata again for categories for the data items. These can be seen in the *categories* section.
![data categories](./images/gaidd48.png?raw=true "Data categories")

4. Go ahead and run this program using the command shown below. This will take the document strings, create embeddings and vectors for them in the Chroma database section and then wait for us to enter a query.
```
python vectordb.py
```
![waiting for input](./images/gaidd49.png?raw=true "Waiting for input")

5. You can enter a query here about any topic and the vector database functionality will try to find the most similar matching data that it has. Since we've only given it a set of 10 strings to work from, the results may not be relevant or very good, but represent the best similarity match the system could find based on the query. Go ahead and enter a query. Some sample ones are shown below, but you can choose others if you want. Just remember it will only be able to choose from the data we gave it. The output will show the closest match from the doc strings and also the similarity and category.
```
Tell me about food.
Who is the most famous person?
How can I learn better?
```
![query results](./images/gaidd50.png?raw=true "Query results")

6. After you've entered and run your query, you can add another one or just type *exit* to stop.

7. Now, let's update the number of results that are returned so we can query on multiple topics. In the file *vectordb.py*, change line 70 to say *n_results=3,* instead of *n_results=1,*. Make sure to save your changes afterwards.

![changed number of results](./images/gaidd51.png?raw=true "Changed number of results")

8. Run the program again with *python vectordb.py*. Now you can try more complex queries or try multiple queries (separated by commas). 

![multiple queries](./images/gaidd52.png?raw=true "Multiple queries")
 
9. When done querying the data, if you have more time, you can try modifying or adding to the document strings in the file, then save your changes and run the program again with queries more in-line with the data you provided. You can type in "exit" for the query to end the program.

10. In preparation for the next lab, remove the *llava* model and download the *llama3.2* model.
```
ollama rm llava
ollama pull llama3.2
```

<p align="center">
**[END OF LAB]**
</p>
</br></br>

    
**Lab 5 - Using RAG with Agents**

**Purpose: In this lab, we’ll explore how agents can leverage external data stores via RAG**

1. For this lab, we have an application that does the following:

- Reads, processes, and stores information about company offices from a PDF file
- Lets you input a starting location
- Lets you prompt about a destination location such as an office name
- Maps the destination back to data taken from the PDF if it can
- Uses the destination from the PDF data or from the prompt to  
  - Find and provide 3 interesting facts about the destination
  - Calculate distance from the starting location to the destination
- Stores information about starting location in an external file
- Repeats until user enters *exit*

2. The PDF file we're using to illustrate RAG here is a fictional list of offices and related info for a company. You can see it in the repo at  [**data/offices.pdf**](./data/offices.pdf) 

![Data pdf](./images/aa66.png?raw=true "Data pdf") 


3. As before, we'll use the "view differences and merge" technique to learn about the code we'll be working with. The command to run this time is below. The code differences mainly hightlight the changes for RAG use in the agent, including working with vector database and snippets returned from searching it.
   
```
code -d ../extra/rag_agent.txt rag_agent.py
```
</br></br>

![Code for rag agent](./images/aa65.png?raw=true "Code for rag agent") 


4. When you're done merging, close the tab as usual to save your changes. Now, in a terminal, run the agent with the command below:

```
python rag_agent.py
```

5. You'll see the agent loading up the embedding pieces it needs to store the document in the vector database. After that you can choose to override the default starting location, or leave it on the default. You'll see a *User:* prompt when it is ready for input from you. The agent is geared around you entering a prompt about an office. Try a prompt like one of the ones below about office "names" that are only in the PDF.

```
Tell me about HQ
Tell me about the Southern office
```

6. What you should see after that are some messages that show internal processing, such as the retrieved items from the RAG datastore.  Then the agent will run through the necessary steps like geocoding locations, calculating distance, using the LLM to get interesting facts about the city etc. At the end it will print out facts about the office location, and the city the office is in, as well as the distance to the office.
 
![Running the RAG agent](./images/aa67.png?raw=true "Running the RAG agent") 

7. The stored information about startup location is in a file named *user_starting_location.json* in the same directory if you want to view that.

8. After the initial run, you can try prompts about other offices or cities mentioned in the PDF. Type *exit* when done.

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 5 - Working with multiple agents**

**Purpose: In this lab, we’ll see how to add an agent to a workflow using CrewAI.**

1. As we've done before, we'll build out the agent code with the diff/merge facility. Run the command below.
```
code -d ../extra/lab5-code.txt agent5.py
```

![Diffs](./images/aa23.png?raw=true "Diffs") 

2. In the *agent5.py* template, we have the imports and llm setup at the top filled in, along with a simulated function to book a flight. Scroll to the bottom. At the bottom is the input and code to kick off the "*crew*". So, we need to fill in the different tasks and setup the crew.

3. Scroll back to the top, review each change and then merge each one in. Notice the occurrences of "*booking_agent*". This is all being done with a single agent in the crew currently. When done, the files should show no differences. Click on the "X" in the tab at the top to save your changes to *agent5.py*.

![Merge complete](./images/aa24.png?raw=true "Merge complete") 

4. Now you can run the agent and see the larger workflow being handled. There will be quite a bit of output so this may take a while to run. **NOTE: Even though the agent may prompt for human input to select a flight, none is needed. We're not adding that in and using fake info to keep things simple and quick.**

```
python agent5.py
```

![Execution](./images/aa31.png?raw=true "Execution") 

5. Now, that we know how the code works and that it works, let's consider the overall approach. Since there are multiple functions going on here (getting info, finding flights, booking flights) it doesn't necessarily make sense to have just one agent doing all those things. Let's add two other agents - a *travel agent* to help with finding flights, and a customer_service_agent to help with user interactions. To start, replace the single *booking agent* definition with these definitions for the 3 agents (making sure to get the indenting correct):

**Directions:** Copy the block of replacement text in gray below and paste over the single agent definition in the code. Reminder - you may need to use keyboard shortcuts to copy and paste. The screenshots are only to show you before and after - they are not what you copy.

```
# Defines the AI agents

booking_agent = Agent(
    role="Airline Booking Assistant",
    goal="Help users book flights efficiently.",
    backstory="You are an expert airline booking assistant, providing the best booking options with clear information.",
    verbose=True,
    llm=ollama_llm,
)

# New agent for travel planning tasks
travel_agent = Agent(
    role="Travel Assistant",
    goal="Assist in planning and organizing travel details.",
    backstory="You are skilled at planning and organizing travel itineraries efficiently.",
    verbose=True,
    llm=ollama_llm,
)

# New agent for customer service tasks
customer_service_agent = Agent(
    role="Customer Service Representative",
    goal="Provide excellent customer service by handling user requests and presenting options.",
    backstory="You are skilled at providing customer support and ensuring user satisfaction.",
    verbose=True,
    llm=ollama_llm,
)
```
![Text to replace](./images/aa26.png?raw=true "Text to replace") 

![Replaced text](./images/aa27.png?raw=true "Replaced text")

6. Next, we'll change each *task definition* to reflect which agent should own it. The places to make the change are in the task definitions in the lines that start with "*agent=*". Just edit each one as needed per the mapping in the table below.

| **Task** | *Agent* | 
| :--------- | :-------- | 
| **extract_travel_info_task** |  *customer_service_agent*  |        
| **find_flights_task** |  *travel_agent*  |  
| **present_flights_task** |  *customer_service_agent*  |  
| **book_flight_task** | *booking_agent* (ok as-is) |  
         
![Replaced text](./images/aa28.png?raw=true "Replaced text")

7. Finally, we need to add the new agents to our crew. Edit the "*agents=[*" line in the block under the comment "*# Create the crew*". In that line, add *customer_service_agent* and *travel_agent*. The full line is below. The screenshot shows the changes made.

```
agents=[booking_agent, customer_service_agent, travel_agent],
```

![Replaced text](./images/aa29.png?raw=true "Replaced text")

8. Now you can save your changes and then run the program again.

```
python agent5.py
```

9. This time when the code runs, you should see the different agents being used in the processing.

![Run with new agents](./images/aa30.png?raw=true "Run with new agents")

<p align="center">
**[END OF LAB]**
</p>
</br></br>
 

<p align="center">
**THANKS!**
</p>
