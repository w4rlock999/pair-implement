# PAIR Algorithm Implementation

This repository contains a demonstration of the PAIR algorithm. The PAIR algorithm is a state-of-the-art jailbreaking algorithm to get objectionable and morally questionable content from a target LLM. It works by pitting another attacker LLM against the target, improving its own jailbreak approach with each attempt.

Follow the instructions below to test the algorithm on your own.

---

## Prerequisites

- **Python Version**: Ensure that you have Python 3.10 or newer installed. Check your Python version by running:

  ```bash
  python --version

---

## Installation Instructions

1. Clone the Repository
  ```bash
  git clone https://github.com/jeffmeredith/pair-implement.git
  ```

2. Navigate to the Project Directory
  ```bash
  cd pair-implement
  ```

3. Create a Virtual Environment
  ```bash
  python -m venv venv
  ```
Next, activate this environment by running:
  ```bash
  .\venv\Scripts\activate
  ```

4. Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```

---

## Set Up OpenAI API Key

5. Obtain an API Key
    1. Go to the [OpenAI API Key page](https://platform.openai.com/api-keys).
    2. Generate a new API key or use an existing one to copy to your clipboard.

6. Set the API Key as an Environment Variable
To allow the script to use your API key, set it as an environment variable.
- Search for "Environment Variables" in the Start menu
- Click "Edit the system environment variables"
- In the System Properties window, click "Environment Variables"
- Under "User variables," click "New" and set:
    - Variable name: ```OPENAI_API_KEY```
    - Variable value: Paste your copied API key

---

## Running the PAIR Algorithm Demo

7. Customize the Attack Objective <br>
Open the ```pair.py``` file in a text editor and locate the line that sets the ```attack_objective``` variable. Modify the string in ```attack_objective``` to match your desired objective.

8. Run the Script <br>
Run the ```pair.py``` script:
  ```bash
  python pair.py
  ```

---

## View Outputs and Chat History

After running the PAIR demo, you can view the resulting jailbreak prompt and chat history between the LLMs in the ```output.txt``` file within the main project directory.