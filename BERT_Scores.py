import nltk
from llama_index.llms.ollama import Ollama
from nltk.tokenize import sent_tokenize
from bert_score import score
llm = Ollama(model="gemma2", request_timeout=1000.0)

def summarize_report_from_file(file_path):
    file=open(file_path, 'r')
    report_text = file.read()
    # Create the prompt for summarization
    prompting = f"""Summarize the following report:\n\n{report_text}"""    
    # Get the summary from the Ollama model
    response = llm.complete(prompt=prompting)
    summary_text = response.text.strip()
    return summary_text

def evaluate_summary_with_bertscore(report, summary):
    P, R, F1 = score([summary], [report], lang="en", verbose=True)
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

# Input path for the report file
file_path = input("Enter the path to the report file: ")
summary = summarize_report_from_file(file_path)
file=open(file_path, 'r')
original_report = file.read()
# Evaluate the summary against the original report using BERT Score
bertscore_result = evaluate_summary_with_bertscore(original_report, summary)
print("\nGenerated Summary:")
print(summary)
print("\nBERTScore Results:")
print(f"Precision: {bertscore_result['Precision']:.3f}")
print(f"Recall: {bertscore_result['Recall']:.3f}")
print(f"F1-Score: {bertscore_result['F1']:.3f}")
