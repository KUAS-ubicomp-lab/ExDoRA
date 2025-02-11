import os

import torch

from exdora_utils import load_model_and_tokenizer, load_data, extract_section

current_dir = os.path.dirname(os.path.abspath(__file__))


def generate_treatment_explanations(clinical_note, in_context_demonstrations, max_length):
    model, tokenizer, device = load_model_and_tokenizer(os.path.join(current_dir, 'plm', 'Asclepius-Mistral-7B-v0.3'))
    prompt = "Below are examples with hospital discharge summaries:\n"
    prompt += f"Example: {in_context_demonstrations}\n"
    prompt += f"Now, analyze the following clinical note and provide a detailed explanation\n\n"
    prompt += f"Patient Case:\n{clinical_note}\nRecommended Treatment and Explanation:"

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
    Social History: The patient is married and worked as a clinical psychologist. Her husband is a pediatric neurologist They have several children, one of which is a nurse.
    Family History: (+) FHx CAD; Father with an MI in his 40's, died
    Brief Hospital Course:
    82 y/o female admitted [**2119-5-4**] for consideration of tracheoplasty. Bronchoscopy done [**5-4**] confirming severe TBM. Underwent
    tracheoplasty [**5-5**], complicated by resp failure d/t mucous plugging, hypoxia requiring re-intubation resulting in prolonged
    ICU and hospital course. Also developed right upper extrem DVT from mid line.
    """

    input_data_list = list(load_data(source='input_data', text='HADM_ID', label_1='CATEGORY', label_2='TEXT').values())[0]
    clinical_note = extract_section(text=input_data_list[2][1], section_name="HISTORY OF PRESENT ILLNESS:")
    treatment_explanations = generate_treatment_explanations(clinical_note, in_context_demonstrations, max_length=5000)
    print("Treatment Explanations:\n", treatment_explanations)


if __name__ == '__main__':
    main()
