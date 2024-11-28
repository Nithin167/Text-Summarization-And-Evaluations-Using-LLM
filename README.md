# Text-Summarization-And-Evaluations-Using-LLM
Text Summarization Using AI

1. Run the following commands in your terminal or command prompt to install the required Python libraries:
   pip install openai
   pip install llama-index
   pip install rouge-score
   pip install bert-score
   pip install nltk
2. Go to the Ollama website and download the appropriate installer for your operating system:
   Once Ollama is installed, open your terminal or command prompt and run these lines to install the necessary models,
   ollama run gemma2
   ollama run mistral
   ollama pull llama3
3. Download the python code files that is attached with this:
   then run the codes and input the report file to summarize it and get a the evaluation metrices.
4. Compare the required summary evaluation metrices for each model and we may obtain which model gives a good summary for a report.
5. Sample Output References for the txt file attached(Himalayas.txt):

MODELS	  ROGUE-1 (Precession, Recall, F1)	ROGUE-2 (Precession, Recall, F1)	ROGUE-L (Precession, Recall, F1)	BERT (Precession, Recall, F1)	  BLEU
Mistral	  0.967, 0.174, 0.296	              0.621, 0.112, 0.189	              0.702, 0.127, 0.215	              0.880, 0.859, 0.870	            0.033
Llama3	  0.947, 0.214, 0.349	              0.738, 0.167, 0.272	              0.784, 0.177, 0.289	              0.880, 0.869, 0.874	            0.015
Gemma2	  0.852, 0.122, 0.213	              0.369, 0.053, 0.092	              0.574, 0.082, 0.144	              0.855, 0.872, 0.863	            0.006

