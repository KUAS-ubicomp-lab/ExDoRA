import torch
from bert_score import BERTScorer
from rouge_score import rouge_scorer

from bart_scorer import BARTScorer
from exdora_utils import average_scores, load_data


def rouge_score(ranked_explanations):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    generated_texts = [i[0] for i in ranked_explanations]
    reference_texts = list(load_data(source='expert_evaluation', text='query', label_1='gpt-3.5-turbo').items())

    for gen_text, ref_text in zip(generated_texts, reference_texts[0][1][1]):
        score = scorer.score(gen_text, ref_text)
        rouge_scores.append(score)

    for idx, rouge_score in enumerate(rouge_scores):
        print(f"ROUGE Scores for pair {idx + 1}:")
        print(f"ROUGE-1: {rouge_score['rouge1']}")
        print(f"ROUGE-L: {rouge_score['rougeL']}")

    # Average ROUGE Scores
    avg_rouge1 = average_scores([score['rouge1'].fmeasure for score in rouge_scores])
    avg_rougeL = average_scores([score['rougeL'].fmeasure for score in rouge_scores])
    return avg_rouge1, avg_rougeL


def BART_score(ranked_explanations):
    scorer = BARTScorer(device='cuda' if torch.cuda.is_available() else 'cpu')
    bart_scores = []
    generated_texts = [i[0] for i in ranked_explanations]
    reference_texts = list(load_data(source='expert_evaluation', text='query', label_1='gpt-3.5-turbo').items())

    for gen_text, ref_text in zip(generated_texts, reference_texts[0][1][1]):
        score = scorer.score(gen_text, ref_text)
        bart_scores.append(score)
    print("BARTScores:", bart_scores)

    # Average BART Scores
    avg_bart_score = average_scores(bart_scores)
    return avg_bart_score


def BERT_score(ranked_explanations):
    scorer = BERTScorer(model_type='bert-base-uncased', device='cuda' if torch.cuda.is_available() else 'cpu')
    bert_scores = []
    generated_texts = [i[0] for i in ranked_explanations]
    reference_texts = list(load_data(source='expert_evaluation', text='query', label_1='gpt-3.5-turbo').items())

    for gen_text, ref_text in zip(generated_texts, reference_texts[0][1][1]):
        P, R, F1 = scorer.score([gen_text], [ref_text])
        bert_scores.append(F1.mean().item())
    print("BERTScores:", bert_scores)

    # Average BERT Scores
    avg_bert_score = average_scores(bert_scores)
    return avg_bert_score
