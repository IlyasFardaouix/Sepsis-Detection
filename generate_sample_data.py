"""Generate sample sepsis detection data for testing."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random


def generate_sample_data(
    n_patients: int = 1000,
    sepsis_rate: float = 0.15,
    hours_per_patient: int = 48,
    output_path: str = "data/raw/sepsis_data.csv",
    random_seed: int = 42,
):
    """
    Generate sample sepsis detection data.

    Args:
        n_patients: Number of patients to generate
        sepsis_rate: Proportion of patients with sepsis
        hours_per_patient: Average number of hours of data per patient
        output_path: Path to save the data
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    print(f"Generating sample data for {n_patients} patients...")

    data = []
    base_time = datetime(2023, 1, 1, 0, 0, 0)

    n_sepsis = int(n_patients * sepsis_rate)
    sepsis_patients = set(random.sample(range(n_patients), n_sepsis))

    for patient_id in range(n_patients):
        has_sepsis = patient_id in sepsis_patients

        # Generate variable number of hours per patient
        n_hours = np.random.poisson(hours_per_patient)
        n_hours = max(6, min(n_hours, 168))  # Between 6 and 168 hours

        # Sepsis onset time (if patient has sepsis)
        if has_sepsis:
            sepsis_onset = np.random.randint(n_hours // 2, n_hours)
        else:
            sepsis_onset = n_hours + 1  # Never occurs

        for hour in range(n_hours):
            time_point = base_time + timedelta(hours=hour)

            # Determine if sepsis has occurred at this time point
            sepsis_label = 1 if has_sepsis and hour >= sepsis_onset else 0

            # Generate features
            # Vital signs
            if sepsis_label == 1:
                # Abnormal values for sepsis patients
                temperature = np.random.normal(38.5, 1.0)  # Fever
                heart_rate = np.random.normal(110, 15)  # Tachycardia
                respiratory_rate = np.random.normal(22, 4)  # Tachypnea
                systolic_bp = np.random.normal(95, 10)  # Hypotension
                oxygen_saturation = np.random.normal(92, 3)  # Low SpO2
            else:
                # Normal values
                temperature = np.random.normal(36.8, 0.5)
                heart_rate = np.random.normal(75, 10)
                respiratory_rate = np.random.normal(16, 2)
                systolic_bp = np.random.normal(120, 10)
                oxygen_saturation = np.random.normal(98, 1)

            # Lab values
            if sepsis_label == 1:
                wbc_count = np.random.normal(15, 4)  # Elevated WBC
                lactate = np.random.normal(3.5, 1.0)  # Elevated lactate
                creatinine = np.random.normal(1.8, 0.5)  # Elevated creatinine
                bilirubin = np.random.normal(2.5, 1.0)  # Elevated bilirubin
            else:
                wbc_count = np.random.normal(7, 2)
                lactate = np.random.normal(1.0, 0.3)
                creatinine = np.random.normal(0.9, 0.2)
                bilirubin = np.random.normal(0.8, 0.3)

            # Add some noise
            temperature = max(35.0, min(42.0, temperature + np.random.normal(0, 0.2)))
            heart_rate = max(40, min(200, heart_rate + np.random.normal(0, 3)))
            respiratory_rate = max(
                8, min(40, respiratory_rate + np.random.normal(0, 1))
            )
            systolic_bp = max(60, min(200, systolic_bp + np.random.normal(0, 2)))
            oxygen_saturation = max(
                70, min(100, oxygen_saturation + np.random.normal(0, 0.5))
            )
            wbc_count = max(2, min(30, wbc_count + np.random.normal(0, 0.5)))
            lactate = max(0.5, min(10, lactate + np.random.normal(0, 0.2)))
            creatinine = max(0.5, min(5, creatinine + np.random.normal(0, 0.1)))
            bilirubin = max(0.2, min(10, bilirubin + np.random.normal(0, 0.2)))

            # Static features
            age = np.random.randint(18, 90)
            gender = np.random.choice([0, 1])  # 0 = Female, 1 = Male
            icu_admission = np.random.choice([0, 1], p=[0.7, 0.3])

            # Time since admission
            hours_since_admission = hour

            data.append(
                {
                    "patient_id": patient_id,
                    "time": time_point,
                    "age": age,
                    "gender": gender,
                    "icu_admission": icu_admission,
                    "hours_since_admission": hours_since_admission,
                    "temperature": round(temperature, 2),
                    "heart_rate": round(heart_rate, 0),
                    "respiratory_rate": round(respiratory_rate, 0),
                    "systolic_bp": round(systolic_bp, 0),
                    "oxygen_saturation": round(oxygen_saturation, 1),
                    "wbc_count": round(wbc_count, 2),
                    "lactate": round(lactate, 2),
                    "creatinine": round(creatinine, 2),
                    "bilirubin": round(bilirubin, 2),
                    "sepsis_label": sepsis_label,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSample data generated successfully!")
    print(f"Total records: {len(df)}")
    print(f"Number of patients: {df['patient_id'].nunique()}")
    print(f"Sepsis cases: {df['sepsis_label'].sum()}")
    print(f"Sepsis rate: {df['sepsis_label'].mean():.2%}")
    print(f"\nData saved to: {output_path}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData summary:")
    print(df.describe())


if __name__ == "__main__":
    generate_sample_data(
        n_patients=1000,
        sepsis_rate=0.15,
        hours_per_patient=48,
        output_path="data/raw/sepsis_data.csv",
        random_seed=42,
    )
