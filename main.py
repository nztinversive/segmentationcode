### This file sets up a conversational agent that uses OpenAI's language model and Meta's SAM image segmentation to describe the appearance of Waldo in a given image. 
### The user inputs a prompt asking to find Waldo and the agent provides a response with a description of his appearance.


import os
import pathlib
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from gradio_tools.tools import SAMImageSegmentationTool
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)

os.environ['OPENAI_API_KEY']

# Initialize a ConversationBufferMemory object to store the chat history
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize a list of tools to use in the agent
tools = [SAMImageSegmentationTool().langchain]

# Specify the path to the Waldo image
waldo = pathlib.Path(__file__).parent / "dog.png"

# Initialize the agent using the specified tools and memory, and set verbose=True to print debug messages
agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)

# Prompt the user to find Waldo in the image and provide a description of his appearance
output = agent.run(input=(f"Please find the dog in this image: {waldo}. "))

# Print the agent's response
print(output)

# Save the observation photo
observation_path = pathlib.Path(__file__).parent / "observation.png"
if '/tmp/' in output:
    observation_file_url = output.split(' ')[-1]
    with open(observation_file_url, 'rb') as original_file:
        file_content = original_file.read()
    with open(observation_path, 'wb') as observation_file:
        observation_file.write(file_content)
    print(f"Observation photo saved as {observation_path}")
else:
    print("Observation photo not found.")

# Save the temporary file with the found Waldo
temp_file_path = pathlib.Path(__file__).parent / "temp_waldo.png"
if 'Here is the image with Waldo identified.' in output:
    temp_file_url = output.split(' ')[-1]
    with open(temp_file_url, 'rb') as original_file:
        file_content = original_file.read()
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(file_content)
    print(f"Temporary file with found Waldo saved as {temp_file_path}")
else:
    print("Waldo not found. No temporary file saved.")

