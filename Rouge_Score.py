from llama_index.llms.ollama import Ollama
from rouge_score import rouge_scorer
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

def evaluate_summary_with_rouge(report, summary):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(report, summary)
    return scores

# Input path for the report file
file_path = input("Enter the path to the report file: ")
summary = summarize_report_from_file(file_path)
file=open(file_path, 'r')
original_report = file.read()
# Evaluate the summary against the original report using Rouge Score
scores = evaluate_summary_with_rouge(original_report, summary)
print("\nGenerated Summary:")
print(summary)
print("\nROUGE Scores:")
print(f"ROUGE-1: Precision: {scores['rouge1'].precision:.3f}, Recall: {scores['rouge1'].recall:.3f}, F1: {scores['rouge1'].fmeasure:.3f}")
print(f"ROUGE-2: Precision: {scores['rouge2'].precision:.3f}, Recall: {scores['rouge2'].recall:.3f}, F1: {scores['rouge2'].fmeasure:.3f}")
print(f"ROUGE-L: Precision: {scores['rougeL'].precision:.3f}, Recall: {scores['rougeL'].recall:.3f}, F1: {scores['rougeL'].fmeasure:.3f}")
