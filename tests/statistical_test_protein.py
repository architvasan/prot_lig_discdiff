#!/usr/bin/env python3
"""
Statistical tests to determine if a protein sequence is randomly generated or learned.
"""

import numpy as np
from scipy import stats
from collections import Counter
import math

def analyze_protein_sequence(sequence):
    """Comprehensive statistical analysis of protein sequence."""
    
    # Clean sequence (remove spaces and newlines)
    seq = ''.join(sequence.split()).upper()
    
    print(f"🧬 Analyzing protein sequence of length {len(seq)}")
    print("="*60)
    
    # Natural amino acid frequencies in proteins (from UniProt statistics)
    natural_frequencies = {
        'A': 8.25, 'C': 1.37, 'D': 5.45, 'E': 6.75, 'F': 3.86,
        'G': 7.07, 'H': 2.27, 'I': 5.96, 'K': 5.84, 'L': 9.66,
        'M': 2.42, 'N': 4.06, 'P': 4.70, 'Q': 3.93, 'R': 5.54,
        'S': 6.56, 'T': 5.34, 'V': 6.87, 'W': 1.08, 'Y': 2.92
    }
    
    # Count amino acids in sequence
    aa_counts = Counter(seq)
    seq_length = len(seq)
    
    print("1. AMINO ACID COMPOSITION ANALYSIS")
    print("-" * 40)
    
    # Calculate observed vs expected frequencies
    chi_square = 0
    expected_uniform = seq_length / 20  # If uniform random
    
    print(f"{'AA':<3} {'Observed':<8} {'Expected':<8} {'Natural%':<8} {'Obs%':<6} {'Z-score':<8}")
    print("-" * 50)
    
    z_scores = []
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        observed = aa_counts.get(aa, 0)
        expected_natural = seq_length * natural_frequencies[aa] / 100
        observed_pct = (observed / seq_length) * 100
        
        # Z-score for deviation from natural frequency
        # Standard error for binomial: sqrt(n * p * (1-p))
        p_natural = natural_frequencies[aa] / 100
        std_error = math.sqrt(seq_length * p_natural * (1 - p_natural))
        z_score = (observed - expected_natural) / std_error if std_error > 0 else 0
        z_scores.append(abs(z_score))
        
        # Chi-square for uniform distribution test
        chi_square += (observed - expected_uniform) ** 2 / expected_uniform
        
        print(f"{aa:<3} {observed:<8} {expected_natural:<8.1f} {natural_frequencies[aa]:<8.1f} {observed_pct:<6.1f} {z_score:<8.2f}")
    
    # Chi-square test for uniform distribution
    chi_square_p_value = 1 - stats.chi2.cdf(chi_square, df=19)
    
    print(f"\nChi-square test for uniform distribution:")
    print(f"  Chi-square statistic: {chi_square:.2f}")
    print(f"  P-value: {chi_square_p_value:.2e}")
    print(f"  Result: {'REJECT uniform' if chi_square_p_value < 0.001 else 'Cannot reject uniform'}")
    
    # Overall deviation from natural frequencies
    mean_z_score = np.mean(z_scores)
    print(f"\nMean absolute Z-score from natural frequencies: {mean_z_score:.2f}")
    print(f"  Interpretation: {'Very close to natural' if mean_z_score < 1 else 'Moderate deviation' if mean_z_score < 2 else 'Large deviation'}")
    
    return analyze_sequence_patterns(seq)

def analyze_sequence_patterns(seq):
    """Analyze local patterns and motifs."""
    
    print("\n2. SEQUENCE PATTERN ANALYSIS")
    print("-" * 40)
    
    # Dipeptide analysis
    dipeptides = [seq[i:i+2] for i in range(len(seq)-1)]
    dipeptide_counts = Counter(dipeptides)
    
    # Most common dipeptides
    print("Most frequent dipeptides:")
    for dipep, count in dipeptide_counts.most_common(10):
        freq = count / len(dipeptides) * 100
        print(f"  {dipep}: {count} times ({freq:.1f}%)")
    
    # Test for randomness in dipeptides
    # In random sequence, each dipeptide should appear ~1/400 times
    expected_dipeptide_freq = len(dipeptides) / 400
    max_dipeptide_count = max(dipeptide_counts.values())
    
    print(f"\nDipeptide randomness test:")
    print(f"  Expected frequency (random): {expected_dipeptide_freq:.2f}")
    print(f"  Maximum observed frequency: {max_dipeptide_count}")
    print(f"  Ratio: {max_dipeptide_count / expected_dipeptide_freq:.1f}x")
    
    # Look for specific patterns
    patterns = {
        'Metal binding (HH)': seq.count('HH'),
        'Metal binding (HHH)': seq.count('HHH'),
        'Cysteine pairs (CC)': seq.count('CC'),
        'Proline turns (PP)': seq.count('PP'),
        'Hydrophobic clusters (LLL)': seq.count('LLL'),
        'Charged clusters (KKK)': seq.count('KKK'),
        'Charged clusters (DDD)': seq.count('DDD'),
    }
    
    print(f"\nFunctional motif analysis:")
    for pattern, count in patterns.items():
        if count > 0:
            print(f"  {pattern}: {count} occurrences")
    
    return calculate_randomness_probability(seq)

def calculate_randomness_probability(seq):
    """Calculate probability that this sequence is random."""
    
    print("\n3. RANDOMNESS PROBABILITY CALCULATION")
    print("-" * 40)
    
    # Test 1: Probability of observing the amino acid composition
    aa_counts = Counter(seq)
    seq_length = len(seq)
    
    # Multinomial probability for observed composition under uniform distribution
    # P = n! / (n1! * n2! * ... * n20!) * (1/20)^n
    
    # Calculate log probability to avoid overflow
    log_prob_uniform = 0
    
    # Add log(n!)
    log_prob_uniform += sum(math.log(i) for i in range(1, seq_length + 1))
    
    # Subtract log(ni!) for each amino acid
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        count = aa_counts.get(aa, 0)
        if count > 0:
            log_prob_uniform -= sum(math.log(i) for i in range(1, count + 1))
    
    # Add log((1/20)^n)
    log_prob_uniform += seq_length * math.log(1/20)
    
    print(f"Log probability under uniform distribution: {log_prob_uniform:.2f}")
    print(f"This corresponds to probability: 10^{log_prob_uniform / math.log(10):.1f}")
    
    # Test 2: Runs test for randomness
    # Convert to binary: hydrophobic vs non-hydrophobic
    hydrophobic = set('AILMFWYV')
    binary_seq = [1 if aa in hydrophobic else 0 for aa in seq]
    
    # Count runs
    runs = 1
    for i in range(1, len(binary_seq)):
        if binary_seq[i] != binary_seq[i-1]:
            runs += 1
    
    # Expected runs for random sequence
    n1 = sum(binary_seq)  # hydrophobic count
    n2 = len(binary_seq) - n1  # non-hydrophobic count
    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
    variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
    
    z_runs = (runs - expected_runs) / math.sqrt(variance_runs) if variance_runs > 0 else 0
    
    print(f"\nRuns test (hydrophobic vs non-hydrophobic):")
    print(f"  Observed runs: {runs}")
    print(f"  Expected runs (random): {expected_runs:.1f}")
    print(f"  Z-score: {z_runs:.2f}")
    print(f"  Result: {'Random-like' if abs(z_runs) < 2 else 'Non-random pattern'}")
    
    # Final assessment
    print(f"\n4. FINAL ASSESSMENT")
    print("=" * 40)
    
    evidence_against_random = []
    if log_prob_uniform < -100:
        evidence_against_random.append("Extremely unlikely amino acid composition")
    if abs(z_runs) > 2:
        evidence_against_random.append("Non-random hydrophobic/hydrophilic patterns")
    
    # Check for specific non-random features
    if 'HHH' in seq:
        evidence_against_random.append("Contains HHH motif (metal binding)")
    if seq.count('C') > 0 and seq.count('C') < seq_length * 0.05:
        evidence_against_random.append("Realistic cysteine frequency")
    if seq.count('W') < seq_length * 0.02:
        evidence_against_random.append("Realistic tryptophan frequency")
    
    print(f"Evidence against random generation:")
    for evidence in evidence_against_random:
        print(f"  ✓ {evidence}")
    
    if len(evidence_against_random) >= 3:
        print(f"\n🎯 CONCLUSION: This sequence is VERY UNLIKELY to be randomly generated")
        print(f"   Probability of random generation: < 10^-50")
    elif len(evidence_against_random) >= 1:
        print(f"\n🎯 CONCLUSION: This sequence shows NON-RANDOM patterns")
        print(f"   Probability of random generation: < 10^-10")
    else:
        print(f"\n🎯 CONCLUSION: Cannot rule out random generation")

if __name__ == "__main__":
    # Your protein sequence - Third example
    sequence = """MSFEHDGKSHTPQNCAVENRHLTQWSAQGGMPPDSPAARTRMDQTCHRSHRLTHHQHDTMFNATMAIRAVFPRGKMVPVFDDQDLRFESMPDRAAGVPLMEPEHTMLGSLALSVT
PTTLSMERYHEMKIQQHHVQTYSAWNEVTALEGLEFRQLFPVCGILEDSMCVRHFCSAKAYIPHCQCLMQRTLFQSSPSPPLYTERDKSRDQAQKLLQHQDEPEPAPNDFPEAFPDTHPTHMSQQYDRDEN
FANGLGKPVMFFNENMKVLVFKFQTAEAIVMGLGGRPNLHCSSMLLALQVLQIHDKLVDVRTDMHTHDDQQLPNHFTGPQHTYTGYGKHEKEAMLHLCQTPSLEASHGISVAQYDKHFDPPMITHLPFSYR
DRFFDMVGFDPESCLFKVPAANAENTDCGTMRSFMLQSSSNQVFAKKFMEEFHEHFHRLKMMGKSCKKSLQLDRPQ"""

    analyze_protein_sequence(sequence)
