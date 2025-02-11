import os

import torch

from exdora_utils import load_model_and_tokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))


def extract_information(discharge_summary):
    history_keywords = ["COPD", "progressive difficulties breathing", "hypoxemia"]
    medications_keywords = ["prilosec 20, mucinex 600, synthroid 75"]
    diagnoses_keywords = ["CARDIOTHORACIC", "COPD", "hypoxemia"]

    extracted_info = {
        "history": [kw for kw in history_keywords if kw in discharge_summary],
        "medications": [kw for kw in medications_keywords if kw in discharge_summary],
        "diagnoses": [kw for kw in diagnoses_keywords if kw in discharge_summary]
    }
    return extracted_info


def generate_treatment_recommendations(extracted_info, engine, max_length, treatment_explanations=None):
    model, tokenizer, device = load_model_and_tokenizer(engine)
    prompt = f"""
    The patient has the following medical history: {', '.join(extracted_info['history'])}.
    Current medications include: {', '.join(extracted_info['medications'])}.
    The patient is diagnosed with: {', '.join(extracted_info['diagnoses'])}.
    Given the following explanation: {treatment_explanations}\nPlease recommend appropriate treatments for the patient.
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    recommendations = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recommendations


def main():
    # Choose model: 'plm/Asclepius-Mistral-7B-v0.3' or 'plm/clinicalmamba-2.8b-hf'
    engine = os.path.join(current_dir, 'plm', 'Asclepius-Mistral-7B-v0.3')
    discharge_summary = """ 
    Medical History: The patient is married and worked as a clinical psychologist. 
    Her husband is a pediatric neurologist They have several children, one of which is a nurse. 
    FHx CAD; Father with an MI in his 40's, died
    Current Medications: prilosec 20, mucinex 600, synthroid 75.
    Clinical Notes: There was significant malacia of the peripheral and central airways with complete collapse of the 
    airways on coughing and forced expiration. .
    """
    extracted_info = extract_information(discharge_summary)
    treatment_recommendations = generate_treatment_recommendations(
        extracted_info=extracted_info,
        engine=engine,
        max_length=5000)
    print("Treatment Recommendations:\n", treatment_recommendations)


if __name__ == '__main__':
    main()
