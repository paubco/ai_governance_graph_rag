"""
Quick test script for semantic chunker
Run this to validate chunking behavior before processing all documents
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.processing.chunking.semantic_chunker import SemanticChunker


def test_chunker():
    """Test chunker with sample regulatory text"""
    
    # Sample text with different sections and topics
    sample_text = """
# Proposed Regulatory Framework for Modifications to Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD)

Discussion Paper and Request for Feedback

![](images/6eb899d522e0620c31df3ccdfbd2aed7f403738e4bd519f3c5cbd61c220aa38e.jpg)

# Proposed Regulatory Framework for Modifications to Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) - Discussion Paper and Request for Feedback

# I. Introduction

Artificial intelligence (AI)- and machine learning (ML)-based technologies have the potential to transform healthcare by deriving new and important insights from the vast amount of data generated during the delivery of healthcare every day. Example high-value applications include earlier disease detection, more accurate diagnosis, identification of new observations or patterns on human physiology, and development of personalized diagnostics and therapeutics. One of the greatest benefits of AI/ML in software resides in its ability to learn from real-world use and experience, and its capability to improve its performance. The ability for AI/ML software to learn from real-world feedback (training) and improve its performance (adaptation) makes these technologies uniquely situated among software as a medical device  $(\mathsf{SaMD})^{1}$  and a rapidly expanding area of research and development. Our vision is that with appropriately tailored regulatory oversight, AI/ML-based SaMD will deliver safe and effective software functionality that improves the quality of care that patients receive.

FDA has made significant strides in developing policies $^{2, 3}$  that are appropriately tailored for SaMD to ensure that safe and effective technology reaches users, including patients and healthcare professionals. Manufacturers submit a marketing application to FDA prior to initial distribution of their medical device, with the submission type and data requirements based on the risk of the SaMD (510(k) notification, De Novo, or premarket approval application (PMA) pathway). For changes in design that are specific to software that has been reviewed and cleared under a 510(k) notification, FDA's Center for Devices and Radiological Health (CDRH) has published guidance (Deciding When to Submit a 510(k) for a Software Change to an Existing Device, $^{4}$  also referred to herein as the software modifications guidance) that describes a risk-based approach to assist in determining when a premarket submission is required. $^{5}$

The International Medical Device Regulators Forum (IMDRF) defines 'Software as a Medical Device (SaMD)' as software intended to be used for one or more medical purposes that perform these purposes without being part of a hardware medical device. FDA, under the Federal Food, Drug, and Cosmetic Act (FD&C Act) considers medical purpose as those purposes that are intended to treat, diagnose, cure, mitigate, or prevent disease or other conditions.

The 510(k) software modifications guidance focuses on the risk to users/patients resulting from the software change. Categories of software modifications that may require a premarket submission include:

- A change that introduces a new risk or modifies an existing risk that could result in significant harm;  
- A change to risk controls to prevent significant harm; and  
- A change that significantly affects clinical functionality or performance specifications of the device.

When applied to AI/ML-based SaMD, the above approach would require a premarket submission to the FDA when the AI/ML software modification significantly affects device performance, or safety and effectiveness<sup>6</sup>; the modification is to the device's intended use; or the modification introduces a major change to the SaMD algorithm. For a PMA-approved SaMD, a supplemental application would be required for changes that affect safety or effectiveness, such as new indications for use, new clinical effects, or significant technology modifications that affect performance characteristics.

To address the critical question of when a continuously learning AI/ML SaMD may require a premarket submission for an algorithm change, we were prompted to reimagine an approach to premarket review for AI/ML-driven software modifications. Such an approach would need to maintain reasonable assurance of safety and effectiveness of AI/ML-based SaMD, while allowing the software to continue to learn and evolve over time to improve patient care.

To date, FDA has cleared or approved several AI/ML-based SaMD. Typically, these have only included algorithms that are "locked" prior to marketing, where algorithm changes likely require FDA premarket review for changes beyond the original market authorization. However, not all AI/ML-based SaMD are locked; some algorithms can adapt over time. The power of these AI/ML-based SaMD lies within the ability to continuously learn, where the adaptation or change to the algorithm is realized after the SaMD is distributed for use and has "learned" from real-world experience. Following distribution, these types of continuously learning and adaptive AI/ML algorithms may provide a different output in comparison to the output initially cleared for a given set of inputs.

The traditional paradigm of medical device regulation was not designed for adaptive AI/ML technologies, which have the potential to adapt and optimize device performance in real-time to continuously improve healthcare for patients. The highly iterative, autonomous, and adaptive nature of these tools requires a new, total product lifecycle (TPLC) regulatory approach that facilitates a rapid cycle of product improvement and allows these devices to continually improve while providing effective safeguards.

This discussion paper proposes a framework for modifications to AI/ML-based SaMD that is based on the internationally harmonized International Medical Device Regulators Forum (IMDRF) risk categorization principles, FDA's benefit-risk framework, risk management principles in the software

modifications guidance $^{8}$ , and the organization-based TPLC approach as envisioned in the Digital Health Software Precertification (Pre-Cert) Program. $^{9}$  It also leverages practices from our current premarket programs, including the 510(k), De Novo, and PMA pathways.

This discussion paper describes an innovative approach that may require additional statutory authority to implement fully. The proposed framework is being issued for discussion purposes only and is not a draft guidance. This document is not intended to communicate FDA's proposed (or final) regulatory expectations but is instead meant to seek early input from groups and individuals outside the Agency prior to development of a draft guidance.

This proposed TPLC approach allows FDA's regulatory oversight to embrace the iterative improvement power of AI/ML SaMD while assuring that patient safety is maintained. It also assures that ongoing algorithm changes are implemented according to pre-specified performance objectives, follow defined algorithm change protocols, utilize a validation process that is committed to improving the performance, safety, and effectiveness of AI/ML software, and include real-world monitoring of performance. This proposed TPLC regulatory framework aims to promote a mechanism for manufacturers to be continually vigilant in maintaining the safety and effectiveness of their SaMD, that ultimately, supports both FDA and manufacturers in providing increased benefits to patients and providers.


"""
    
    print("="*60)
    print("SEMANTIC CHUNKER TEST")
    print("="*60)
    
    # Initialize chunker
    print("\nInitializing chunker...")
    chunker = SemanticChunker(
        threshold=0.7,
        min_sentences=3,
        max_tokens=1500
    )
    
    # Chunk the document
    print("\nChunking document...")
    chunks = chunker.chunk_document(
        text=sample_text,
        document_id="TEST_DOC",
        metadata={'source': 'test', 'type': 'regulation'}
    )
    
    # Display results
    print(f"\nCreated {len(chunks)} chunks:\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"{'─'*60}")
        print(f"CHUNK {i}: {chunk.chunk_id}")
        print(f"Section: {chunk.section_header or 'No header'}")
        print(f"Stats: {chunk.sentence_count} sentences, {chunk.token_count} tokens")
        print(f"\nText preview:")
        print(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)
        print()
    
    # Statistics
    stats = chunker.get_statistics(chunks)
    print("="*60)
    print("STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_chunker()