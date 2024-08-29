import os

import torch

from exdora_utils import load_model_and_tokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))


def generate_treatment_explanations(clinical_note, in_context_demonstrations, max_length):
    model, tokenizer, device = load_model_and_tokenizer(os.path.join(current_dir, 'plm', 'meditron-7b'))
    prompt = f"{in_context_demonstrations}\n\nPatient Case:\n{clinical_note}\nRecommended Treatment and Explanation:"

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
    explanations = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanations


def main():
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

    clinical_note = """
    Patient History: 45-year-old male with a history of major depressive disorder (MDD),
    presenting with fatigue, insomnia, and loss of interest. Previous treatment included SSRIs 
    and psychotherapy with partial response.
    Current Medications: Sertraline 50 mg daily.
    Clinical Notes: Patient reports mild improvement in mood but still experiences significant
    anxiety and persistent insomnia. Recent lab tests show normal thyroid function and no 
    electrolyte abnormalities.
    Diagnoses: Major Depressive Disorder (MDD), Generalized Anxiety Disorder (GAD), Insomnia.
    """
    treatment_explanations = generate_treatment_explanations(clinical_note, in_context_demonstrations, max_length=600)
    print("Treatment Explanations:\n", treatment_explanations)


if __name__ == '__main__':
    main()
