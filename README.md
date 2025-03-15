# Medical Chatbot

A chatbot that provides medical assistance using a model from the Hugging Face API. The chatbot processes user queries and provides relevant medical responses based on pre-trained AI models. It also leverages LangChain to extract content from books and FAISS for efficient similarity search.

## Features
- Uses a Hugging Face model to process medical queries.
- Provides AI-generated responses to health-related questions.
- Integrates LangChain to extract medical context from books.
- Utilizes FAISS for fast and efficient similarity search.
- Built with Python and integrates with a chatbot interface.

## Installation

### Prerequisites
- Python 3.x
- An API key for Hugging Face (if required by the model)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/medical-chatbot.git
   cd medical-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If you don't have a `requirements.txt`, install the dependencies manually:
   ```bash
   pip install requests transformers streamlit langchain faiss-cpu
   ```

## Usage

1. Run the chatbot:
   ```bash
   python bot.py
   ```

2. Interact with the chatbot by entering medical queries.

## API Integration
This chatbot uses a model from Hugging Face. If an API key is needed, set it up as follows:
```bash
export HUGGINGFACE_API_KEY='your_api_key_here'
```

## LangChain and FAISS
- **LangChain** is used to extract relevant medical context from books, enhancing response accuracy.
- **FAISS** is used for similarity search, improving the chatbot's retrieval of relevant medical information.

## Troubleshooting
- Ensure the Hugging Face API is accessible.
- If responses are inaccurate, try fine-tuning the model or using a different pre-trained model.
- Ensure FAISS is correctly installed for optimal performance.

## License
This project is licensed under the MIT License.

## Author
Kartik Verma


