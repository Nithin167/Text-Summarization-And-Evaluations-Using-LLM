import nltk
from llama_index.llms.ollama import Ollama
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
llm = Ollama(model="gemma2", request_timeout=1000.0)

def summarize_report_from_file(file_path):
    # Read the report from the file
    file=open(file_path, 'r')
    report_text = file.read()
    # Create the prompt for summarization
    prompting = f"""Summarize the following report:\n\n{report_text}"""
    # Get the summary from the Ollama model
    response = llm.complete(prompt=prompting)
    summary_text = response.text.strip()
    return summary_text

def evaluate_summary_with_bleu(report, summary):
    # Tokenize the report and summary
    report_tokens = nltk.word_tokenize(report)
    summary_tokens = nltk.word_tokenize(summary)
    # BLEU score requires the reference as a list of lists
    reference = [report_tokens]
    # Calculate BLEU score with smoothing to handle short texts
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, summary_tokens, smoothing_function=smoothie)
    return bleu_score

# Input path for the report file
file_path = input("Enter the path to the report file: ")
summary = summarize_report_from_file(file_path)
file=open(file_path, 'r')
original_report = file.read()
# Evaluate the summary against the original report using BLEU
bleu_score = evaluate_summary_with_bleu(original_report, summary)
print("\nGenerated Summary:")
print(summary)
print("\nBLEU Score:")
print(f"{bleu_score:.3f}")
