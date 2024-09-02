import os

from treatment_explanations_generator import generate_treatment_explanations
from treatment_recommender import extract_information, generate_treatment_recommendations

current_dir = os.path.dirname(os.path.abspath(__file__))


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
    treatment_explanations = generate_treatment_explanations(clinical_note=discharge_summary,
                                                             in_context_demonstrations=in_context_demonstrations,
                                                             max_length=50000)
    treatment_recommendations = generate_treatment_recommendations(extracted_info=extracted_info,
                                                                   engine=engine,
                                                                   max_length=50000,
                                                                   treatment_explanations=treatment_explanations)
    print("Treatment Recommendations with Explanations:\n", treatment_recommendations)


if __name__ == '__main__':
    main()
