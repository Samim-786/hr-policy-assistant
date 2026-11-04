import requests

API_URL = "http://127.0.0.1:8000/ask"


TEST_CASES = [
    {
        "name": "Casual leave carry-forward",
        "question": "How many casual leave days can I carry forward?",
        "expected_status": "answered",
        "expected_terms": ["8 days"],
        "expected_document": "leave-policy.md",
        "expected_section": "4.1 Casual leave carry-forward"
    },
    {
        "name": "Standard dental coverage",
        "question": "Does the Standard health tier cover dental implants?",
        "expected_status": "answered",
        "expected_terms": ["not cover", "not covered"],
        "expected_document": "benefits-policy.md",
        "expected_section": "2. Health coverage tiers"
    },
    {
        "name": "Plus dental coverage",
        "question": "How much dental implant coverage does the Plus tier provide?",
        "expected_status": "answered",
        "expected_terms": ["40000", "40,000"],
        "expected_document": "benefits-policy.md",
        "expected_section": "2. Health coverage tiers"
    },
    {
        "name": "Lost laptop reporting",
        "question": "How quickly should I report a stolen company laptop?",
        "expected_status": "answered",
        "expected_terms": ["4 hours"],
        "expected_document": "it-security-policy.md",
        "expected_section": "3. Devices"
    },
    {
        "name": "Home gym reimbursement",
        "question": "Can I get reimbursement for home gym equipment?",
        "expected_status": "answered",
        "expected_terms": ["not a reimbursable benefit",
                           "not reimbursable "],
        "expected_document": "benefits-policy.md",
        "expected_section": "4. Wellness"
    },
    {
        "name": "Internet reimbursement",
        "question": "Does the company reimburse internet bills?",
        "expected_status": "insufficient_information",
        "expected_terms": [],
        "expected_document": None,
        "expected_section": None
    },
    {
        "name": "Maternity leave",
        "question": "What is the company's maternity leave entitlement?",
        "expected_status": "insufficient_information",
        "expected_terms": [],
        "expected_document": None,
        "expected_section": None
    }
]


def normalize(text):
    return text.lower().replace(",", "").replace("₹", "")


def run_test(test_case):

    response = requests.post(
        API_URL,
        json={"question": test_case["question"]},
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    errors = []

    # Check status
    if result["status"] != test_case["expected_status"]:
        errors.append("Incorrect response status")

    # Check answer content
    answer = normalize(result["answer"])

    expected_terms = test_case["expected_terms"]

    if expected_terms and not any(
        normalize(term) in answer
        for term in expected_terms
    ):
        errors.append("Expected information missing from answer")

    # Check citations
    citations = result.get("citations", [])

    expected_document = test_case["expected_document"]
    expected_section = test_case["expected_section"]

    if expected_document:

        citation_found = any(
            citation["document_name"] == expected_document
            and expected_section in citation["section"]
            for citation in citations
        )

        if not citation_found:
            errors.append("Expected citation missing")

    else:

        if citations:
            errors.append("Unexpected citations in refusal response")

    if errors:
        return False, errors

    return True, []


def main():

    passed = 0
    failed = 0

    print("\nHR POLICY ASSISTANT - EVALUATION\n")

    for test_case in TEST_CASES:

        try:

            success, errors = run_test(test_case)

            if success:
                passed += 1
                print(f"PASS: {test_case['name']}")

            else:
                failed += 1
                print(f"FAIL: {test_case['name']}")

                for error in errors:
                    print(f"    - {error}")

        except Exception as exc:

            failed += 1

            print(f"ERROR: {test_case['name']}")
            print(f"    - {exc}")

    print("\n------------------------------")
    print(f"Total tests: {len(TEST_CASES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    accuracy = (passed / len(TEST_CASES)) * 100

    print(f"Pass rate: {accuracy:.2f}%")
    print("------------------------------")


if __name__ == "__main__":
    main()