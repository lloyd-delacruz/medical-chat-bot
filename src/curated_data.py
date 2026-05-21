"""Curated, current medical reference documents.

Grounded in public guidance from WHO, CDC, and NIH/MedlinePlus. Each entry
carries its source URL and the date it was last verified. These are stable
clinical facts that provide a reproducible, offline baseline of up-to-date
information even when the live web fetcher is disabled.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document

VERIFIED_ON = "2026-05-21"

_ENTRIES = [
    {
        "title": "Stroke warning signs (BE-FAST)",
        "url": "https://www.cdc.gov/stroke/signs-symptoms/",
        "content": (
            "Stroke is a medical emergency. Use BE-FAST to recognize warning signs. "
            "B - Balance: sudden loss of balance or coordination. "
            "E - Eyes: sudden trouble seeing in one or both eyes. "
            "F - Face drooping: ask the person to smile; one side may droop. "
            "A - Arm weakness: ask them to raise both arms; one may drift down. "
            "S - Speech difficulty: speech may be slurred or hard to understand. "
            "T - Time to call emergency services immediately. Other sudden signs "
            "include numbness on one side of the body, confusion, trouble walking or "
            "dizziness, and a severe headache with no known cause. "
            "Note the time symptoms first appeared, because some treatments work best "
            "when given soon after onset."
        ),
    },
    {
        "title": "Normal adult vital sign ranges",
        "url": "https://medlineplus.gov/ency/article/002341.htm",
        "content": (
            "Typical resting vital signs for healthy adults. Body temperature averages "
            "about 98.6 F (37 C), with a normal range of roughly 97.7 to 99.1 F "
            "(36.5 to 37.3 C). Resting heart rate is about 60 to 100 beats per minute. "
            "Respiratory rate is about 12 to 18 breaths per minute at rest. A blood "
            "pressure below 120/80 mm Hg is normal, while 130/80 mm Hg or higher is "
            "considered high. Individual normal values vary with age, fitness, and "
            "health conditions."
        ),
    },
    {
        "title": "When to seek emergency care (adults)",
        "url": "https://medlineplus.gov/ency/article/001927.htm",
        "content": (
            "Seek emergency care for difficulty breathing or shortness of breath; chest "
            "or upper-abdominal pain or pressure lasting two minutes or more; fainting "
            "or sudden dizziness or weakness; sudden changes in vision; confusion or "
            "change in mental status; any sudden or severe pain; uncontrolled bleeding; "
            "severe or persistent vomiting or diarrhea; coughing or vomiting blood; "
            "suicidal feelings; or a head injury with loss of consciousness. When in "
            "doubt about a serious symptom, seek emergency care."
        ),
    },
    {
        "title": "Hypertension (high blood pressure) basics",
        "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "content": (
            "Hypertension is when blood pressure is too high. It is written as systolic "
            "over diastolic; a reading at or above 140/90 mm Hg on two different days is "
            "commonly used to diagnose it. It often has no symptoms, so measurement "
            "matters. Risk factors include older age, family history, excess weight, "
            "physical inactivity, high-salt diets, and excessive alcohol use. It raises "
            "the risk of heart attack, stroke, and kidney disease. Management includes a "
            "healthy diet, less salt, physical activity, avoiding tobacco, limiting "
            "alcohol, and prescribed medicines."
        ),
    },
    {
        "title": "Type 2 diabetes basics",
        "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "content": (
            "Diabetes is a chronic condition with high blood glucose. In type 2 diabetes "
            "the body does not use insulin well. Symptoms can include increased thirst, "
            "frequent urination, increased hunger, fatigue, and blurred vision, though "
            "many people have no early symptoms. Risk factors include excess weight, "
            "physical inactivity, and family history. Over time it can damage the heart, "
            "blood vessels, eyes, kidneys, and nerves. Prevention and management focus on "
            "healthy eating, regular activity, a healthy weight, and prescribed medicines."
        ),
    },
    {
        "title": "Medication safety and interactions (general guidance)",
        "url": "https://medlineplus.gov/druginformation.html",
        "content": (
            "Use medicines safely. Keep an up-to-date list of all prescription drugs, "
            "over-the-counter medicines, vitamins, and supplements, and share it with "
            "your doctor and pharmacist. Interactions can change how a medicine works or "
            "increase side effects. Combining medicines that cause drowsiness (such as "
            "opioids, sedatives, and alcohol) can dangerously slow breathing. Some pain "
            "relievers (NSAIDs such as ibuprofen) can interact with blood-pressure "
            "medicines and blood thinners. Read labels, follow dosing instructions, and "
            "ask a pharmacist before combining medicines. This is general information, "
            "not personalized advice."
        ),
    },
    {
        "title": "Everyday preventive health measures",
        "url": "https://www.cdc.gov/chronic-disease/prevention/index.html",
        "content": (
            "Core preventive steps that lower the risk of many diseases. Do not smoke and "
            "avoid tobacco. Be physically active; adults should aim for at least 150 "
            "minutes of moderate activity per week. Eat a balanced diet rich in "
            "vegetables, fruits, and whole grains while limiting salt, added sugar, and "
            "saturated fat. Maintain a healthy weight, limit alcohol, get recommended "
            "vaccinations and screenings, wash hands regularly, and get adequate sleep "
            "of at least 7 hours per night. Routine check-ups help detect problems early."
        ),
    },
]


def _build() -> List[Document]:
    docs: List[Document] = []
    for entry in _ENTRIES:
        docs.append(
            Document(
                page_content=f"{entry['title']}.\n{entry['content']}",
                metadata={
                    "source": entry["title"],
                    "source_type": "curated",
                    "url": entry["url"],
                    "retrieved_at": VERIFIED_ON,
                },
            )
        )
    return docs


CURATED_DOCS: List[Document] = _build()
