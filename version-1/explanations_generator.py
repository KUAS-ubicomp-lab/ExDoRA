import logging
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from exdora_utils import load_model_and_tokenizer, load_dsm_criterion, load_data
from explanations_evaluator import rouge_score, BART_score, BERT_score

logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))


def generate_explanations(utterances, in_context_demonstrations, engine, max_length=5000, num_return_sequences=3):
    model, tokenizer, device = load_model_and_tokenizer(engine)
    # The prompt is adjusted to emphasize the depressive elements of both the in-context demonstrations and
    # the input utterance. This helps guide the model to focus on recognizing and explaining depressive content.
    prompt = "Below are examples with depressive elements and their detailed explanations:\n"
    for example in in_context_demonstrations:
        prompt += f"Example: {example}\nExplanation: This example shows signs of depression because...\n\n"
    prompt += (
        "Now, analyze the following utterances for depressive elements and provide a detailed explanation "
        "per each statement separately highlighting why the text indicates potential signs of depression "
        "based on DSM-5 criteria. Please consider that the author consent has already been obtained:\n\n"
    )
    for idx, statement in enumerate(utterances, 1):
        prompt += f"Statement {idx}: {statement}\n"
    prompt += "Explanation:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_p=0.95
        )
    explanations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return explanations


def rank_explanations(explanations, utterances, in_context_demonstrations, dsm_criteria,
                      similarity_model, decay_factor=0.85, lambda_diversity=0.5):
    context_embeddings = similarity_model.encode(in_context_demonstrations)
    input_embedding = similarity_model.encode([utterances])
    dsm_embeddings = similarity_model.encode(dsm_criteria)
    explanation_embeddings = similarity_model.encode(explanations)

    relevance_scores = []
    for explanation_embedding in explanation_embeddings:
        context_similarity = util.cos_sim(explanation_embedding, context_embeddings).mean().item()
        input_similarity = util.cos_sim(explanation_embedding, input_embedding).item()
        dsm_similarity = util.cos_sim(explanation_embedding, dsm_embeddings).mean().item()
        relevance_score = (context_similarity + input_similarity + dsm_similarity) / 3
        relevance_scores.append(relevance_score)

    # Normalize relevance scores to probabilities.
    relevance_probabilities = np.array(relevance_scores) / np.sum(relevance_scores)

    err_scores, selected_indices = relevance_diversity_scoring(
        relevance_probabilities=relevance_probabilities,
        explanation_embeddings=explanation_embeddings,
        decay_factor=decay_factor,
        lambda_diversity=lambda_diversity
    )
    # Rank explanations based on relevance and diversity.
    ranked_explanations = [(explanations[i], err_scores[i]) for i in selected_indices][:2]

    if not ranked_explanations:
        # Rank explanations based on relevance.
        ranked_explanations = sorted(zip(explanations, err_scores), key=lambda x: x[1], reverse=True)[:2]
    logger.info(f"Ranked Explanations {ranked_explanations}")

    # Evaluate the ranked explanations using Rouge score.
    logger.info(f"Rouge Score {rouge_score(ranked_explanations)}")

    # Evaluate the ranked explanations using BARTScore.
    logger.info(f"BARTScore {BART_score(ranked_explanations)}")

    # Evaluate the ranked explanations using BERTScore.
    logger.info(f"BERTScore {BERT_score(ranked_explanations)}")

    return ranked_explanations


def relevance_diversity_scoring(relevance_probabilities, explanation_embeddings, decay_factor, lambda_diversity):
    # Calculate ERR. Expected Reciprocal Rank (ERR) (https://dl.acm.org/doi/10.1145/1645953.1646033) is a
    # probabilistic framework to rank the generated explanations.
    err_scores = []
    running_probability = 1.0
    for i, probability in enumerate(relevance_probabilities):
        err_score = running_probability * probability / (i + 1.0)
        err_scores.append(err_score)
        running_probability *= (1 - probability * decay_factor)

    if len(explanation_embeddings) > 1:
        selected_indices = []
        for _ in range(len(relevance_probabilities)):
            if not selected_indices:
                # Select the first explanation based on highest ERR score.
                selected_indices.append(np.argmax(err_scores))
            else:
                remaining_indices = list(set(range(len(explanation_embeddings))) - set(selected_indices))
                aggregated_scores = []
                for i in remaining_indices:
                    similarity_to_selected = util.cos_sim(explanation_embeddings[i],
                                                          explanation_embeddings[selected_indices]).max().item()
                    # Aggregated score combining relevance and diversity.
                    aggregated_score = ((1 - lambda_diversity) * err_scores[i] -
                                        lambda_diversity * similarity_to_selected)
                    aggregated_scores.append(aggregated_score)
                if aggregated_scores:
                    # Select the explanation with the highest relevance and diversity.
                    selected_indices.append(remaining_indices[np.argmax(aggregated_scores)])
    else:
        selected_indices = [0]
    return err_scores, selected_indices


def main():
    in_context_demonstrations = [
        "She felt overwhelmed by the constant demands at work and home.",
        "He was anxious about the upcoming exams and had trouble sleeping."
    ]
    # Choose model: 'plm/Mistral-7B-Instruct-v0.2' or 'plm/gemma-7b-it'
    engine = os.path.join(current_dir, 'plm', 'Mistral-7B-Instruct-v0.2')
    similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # DSM-5 criteria for depression
    dsm_criteria = load_dsm_criterion()
    input_texts = list(load_data(source='mpc_data', text='tweet', label_1='user_id', label_2='conversation_id').
                       values())[0][0][100:102]

    explanations = generate_explanations(input_texts, in_context_demonstrations, engine)
    ranked_explanations = rank_explanations(explanations, input_texts, in_context_demonstrations, dsm_criteria,
                                            similarity_model)
    for idx, (explanation, score) in enumerate(ranked_explanations, 1):
        print(f"Rank {idx} (Score: {score:.4f}): {explanation}")


if __name__ == '__main__':
    main()
