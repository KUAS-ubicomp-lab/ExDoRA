import os

import torch

from treatment_explanations_generator import generate_treatment_explanations
from exdora_utils import load_model_and_tokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))


def extract_information(discharge_summary):
    history_keywords = ["history of major depressive disorder", "fatigue", "insomnia", "loss of interest"]
    medications_keywords = ["Sertraline 50 mg daily"]
    diagnoses_keywords = ["Major Depressive Disorder", "Generalized Anxiety Disorder", "Insomnia"]

    extracted_info = {
        "history": [kw for kw in history_keywords if kw in discharge_summary],
        "medications": [kw for kw in medications_keywords if kw in discharge_summary],
        "diagnoses": [kw for kw in diagnoses_keywords if kw in discharge_summary]
    }
    return extracted_info


def generate_treatment_recommendations(extracted_info, treatment_explanations, engine, max_length):
    model, tokenizer, device = load_model_and_tokenizer(engine)
    prompt = f"""
    The patient has the following medical history: {', '.join(extracted_info['history'])}.
    Current medications include: {', '.join(extracted_info['medications'])}.
    The patient is diagnosed with: {', '.join(extracted_info['diagnoses'])}.
    Given the following explanation:\n{treatment_explanations}\nPlease recommend appropriate treatments for the patient.
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
    Patient History: 45-year-old male with a history of major depressive disorder (MDD),
    presenting with fatigue, insomnia, and loss of interest. Previous treatment included SSRIs 
    and psychotherapy with partial response.
    Current Medications: Sertraline 50 mg daily.
    Clinical Notes: Patient reports mild improvement in mood but still experiences significant
    anxiety and persistent insomnia. Recent lab tests show normal thyroid function and no 
    electrolyte abnormalities.
    Diagnoses: Major Depressive Disorder (MDD), Generalized Anxiety Disorder (GAD), Insomnia.
    """
    in_context_demonstrations = """
        Example 1:
        Patient History: 60-year-old male with hypertension and type 2 diabetes.
        Current Medications: Metformin 1000 mg daily, Lisinopril 20 mg daily.
        Clinical Notes: Blood pressure remains elevated despite treatment.
        Diagnoses: Hypertension, Type 2 Diabetes Mellitus.
        Recommended Treatment: Increase Lisinopril to 40 mg daily for better blood pressure control.
        Explanation: Due to persistent hypertension, increasing the dosage of Lisinopril is recommended to manage the condition more effectively.

        Example 2:
        Patient History: 50-year-old female with a history of breast cancer and anxiety.
        Current Medications: Tamoxifen 20 mg daily, Sertraline 50 mg daily.
        Clinical Notes: The patient reports feeling anxious and difficulty sleeping.
        Diagnoses: Anxiety, Breast Cancer.
        Recommended Treatment: Add Trazodone 50 mg at bedtime for sleep and anxiety management.
        Explanation: Trazodone is commonly used for insomnia and can help with anxiety, making it a suitable adjunct to the patient's current regimen.
        """
    extracted_info = extract_information(discharge_summary)
    treatment_explanations = generate_treatment_explanations(discharge_summary, in_context_demonstrations, max_length=5000)
    treatment_recommendations = generate_treatment_recommendations(extracted_info, treatment_explanations, engine, max_length=5000)
    print("Treatment Recommendations:\n", treatment_recommendations)


if __name__ == '__main__':
    main()
