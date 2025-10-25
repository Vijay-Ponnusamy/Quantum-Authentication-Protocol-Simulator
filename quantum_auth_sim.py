# quantum_auth_sim.py
"""
Quantum Authentication Protocol Simulator (Streamlit)
Supports two simple demo protocols:
  1) BB84 Challenge-Response Authentication
  2) Bell-pair (entanglement) Challenge-Response Authentication

"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from hashlib import sha256
import base64

# ---- Page & dark CSS ----
st.set_page_config(page_title="Quantum Authentication Protocol Simulator", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container, .main, header, .sidebar .sidebar-content {
        background-color: #0b0c0e;
        color: #e6eef7;
    }
    .stButton>button { background-color:#1f6feb; color: white; border-radius: 8px; }
    .stSlider>div>div>div>div { color: #e6eef7; }
    .stTextInput>div>div>input { background-color: #14171a; color: #e6eef7; }
    .stDataFrame table { color: #e6eef7; background-color: #0b0c0e; }
    .element-container img { background-color: #0b0c0e; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Quantum Authentication Protocol Simulator 🔐")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Simulation Settings")
    protocol = st.selectbox("Protocol", ["BB84 Challenge-Response"])
    n_qubits = st.slider("Number of qubits / rounds", 4, 256, 32, step=4)
    noise = st.slider("Channel noise (bit-flip probability)", 0.0, 0.5, 0.05, step=0.01)
    attacker = st.checkbox("Simulate active attacker (intercept-resend)", value=False)
    seed = st.number_input("RNG seed", min_value=0, max_value=2**31 - 1, value=42)
    run_button = st.button("Run Simulation")

RNG = np.random.default_rng(int(seed))

# ---- Utilities ----
def sha_digest_hex(b: bytes) -> str:
    return sha256(b).hexdigest()

def apply_bit_flip(state_vector, wire, prob):
    # not used — we instead simulate noise by flipping measurement outcomes with prob
    return state_vector

# ---- BB84-based challenge-response functions ----
def bb84_prepare_states(n, RNG):
    """Return tuple (bits, bases, states) where:
       bits: array of 0/1 logical values (the encoded bits)
       bases: 0->Z (computational), 1->X (Hadamard) basis
       states: list of single-qubit states prepared as rotation angles for qnode
    """
    bits = RNG.integers(0, 2, size=n)
    bases = RNG.integers(0, 2, size=n)
    # We'll encode as RY rotations: |0> -> angle 0, |1> -> angle pi
    # For X basis, apply H (we simulate by rotating angles accordingly)
    # But when using pennylane qnode we'll explicitly apply H where needed.
    return bits, bases

def bb84_simulate(bits, bases_sent, noise_prob, interceptor=False, RNG=None):
    """Simulate sending the BB84 qubits to Bob; Bob measures in random bases.
       Returns: bases_bob, bits_measured (after noise/interception)
    """
    n = len(bits)
    bases_bob = RNG.integers(0, 2, size=n)
    measured = np.zeros(n, dtype=int)

    # We'll simulate per-qubit: if interceptor True => attacker measures in random basis and resends
    for i in range(n):
        sent_bit = bits[i]
        sent_basis = bases_sent[i]
        # If interceptor, attacker picks a random basis and measures (causes disturbance)
        if interceptor:
            att_basis = RNG.integers(0, 2)
            # attacker measures; if wrong basis, outcome random
            if att_basis == sent_basis:
                att_outcome = sent_bit
            else:
                att_outcome = RNG.integers(0, 2)
            # attacker resends qubit encoded with att_outcome in att_basis
            # Now channel noise may flip again
            effective_bit = att_outcome
            effective_basis = att_basis
        else:
            effective_bit = sent_bit
            effective_basis = sent_basis

        # Channel noise: flip the bit with prob noise_prob (simulates bit-flip during transmission)
        if RNG.random() < noise_prob:
            effective_bit = 1 - effective_bit

        # Bob measures in his basis
        bob_basis = bases_bob[i]
        if bob_basis == effective_basis:
            measured[i] = effective_bit
        else:
            # different basis => measurement random in general
            measured[i] = RNG.integers(0, 2)
    return bases_bob, measured

def bb84_authenticate(n_qubits, noise, interceptor, RNG):
    # Alice chooses a random nonce and encodes it into n_qubits bits/bases
    nonce = RNG.integers(0, 2**30)
    nonce_bytes = int(nonce).to_bytes(4, byteorder='big')
    # We'll embed the nonce bits in the first k bits (or expand the nonce)
    bits_full = RNG.integers(0, 2, size=n_qubits)  # random payload; we'll XOR nonce bits into a segment
    # XOR in nonce bits into the first 16 bits (deterministic placement for demo)
    nonce_bits = np.unpackbits(np.frombuffer(nonce_bytes, dtype=np.uint8))  # 32 bits
    k = min(len(nonce_bits), n_qubits)
    bits_full[:k] ^= nonce_bits[:k]
    bases_alice = RNG.integers(0, 2, size=n_qubits)

    # Simulate transmission
    bases_bob, measured = bb84_simulate(bits_full, bases_alice, noise, interceptor=interceptor, RNG=RNG)

    # Sifting: identify indices where bases matched
    matched_idx = np.where(bases_alice == bases_bob)[0]
    # Authentication test: reveal a random subset of matched indices and compare
    # Take up to test_size indices for verification
    test_size = max(1, int(0.2 * len(matched_idx)))
    test_idx = RNG.choice(matched_idx, size=min(test_size, len(matched_idx)), replace=False) if len(matched_idx)>0 else np.array([], dtype=int)
    # compute mismatch rate on test indices
    mismatches = np.sum(bits_full[test_idx] != measured[test_idx])
    mismatch_rate = mismatches / (len(test_idx) + 1e-12)
    # Authentication decision: require mismatch_rate below threshold
    threshold = 0.1  # allow up to 10% mismatches in test subset
    auth_pass = (mismatch_rate <= threshold) and (len(test_idx) > 0)

    # For challenge-response, remaining matched bits (excluding test indices) become response
    remaining_idx = np.setdiff1d(matched_idx, test_idx)
    response_bits = measured[remaining_idx]  # Bob's version of response
    # We compute digest of response bits as authentication token
    if response_bits.size == 0:
        response_token = ""
    else:
        rb_bytes = np.packbits(response_bits)
        response_token = sha256(rb_bytes.tobytes()).hexdigest()

    return dict(
        nonce=nonce,
        nonce_bits=nonce_bits[:k],
        n_qubits=n_qubits,
        bases_alice=bases_alice,
        bases_bob=bases_bob,
        bits_sent=bits_full,
        measured=measured,
        matched_count=len(matched_idx),
        test_size=len(test_idx),
        mismatches=int(mismatches),
        mismatch_rate=float(mismatch_rate),
        auth_pass=auth_pass,
        response_token=response_token
    )

# ---- Bell-pair entanglement-based protocol (toy) ----
def bell_pair_circuit_simulation(n_pairs, noise, interceptor, RNG):
    """
    Simulate sharing n_pairs Bell pairs. Alice encodes nonce by optionally applying X to her qubits
    at positions where nonce bit=1. Bob measures in Z-basis and compares parity/correlation.
    Attacker (interceptor) can break entanglement by measuring/resending (simulated).
    """
    # Generate random nonce bits (length = n_pairs)
    nonce_bits = RNG.integers(0, 2, size=n_pairs)
    # Start with perfect Bell correlations: measured_bob = measured_alice (mod 2) for |Φ+>
    # We'll simulate measurement outcomes
    measured_alice = np.zeros(n_pairs, dtype=int)
    measured_bob = np.zeros(n_pairs, dtype=int)
    disturbance = 0

    for i in range(n_pairs):
        # ideal Bell pair: correlated bits (same)
        base_outcome = RNG.integers(0, 2)
        # Alice encodes nonce by flipping her qubit (apply X) if nonce_bit==1 -> flips her outcome
        alice_out = base_outcome ^ int(nonce_bits[i] == 1)
        bob_out = base_outcome
        # If attacker intercepts one qubit and measures, may randomize correlation
        if interceptor:
            # attacker measures Alice's sent qubit in Z, resends a state consistent with measurement
            attacker_meas = RNG.integers(0, 2)
            # after attacker action, Bob's correlation to Alice may be lost (attacker measurement breaks entanglement)
            bob_out = RNG.integers(0, 2)
            alice_out = attacker_meas  # attacker-sent outcome to Bob might mismatch
            disturbance += 1
            # If nonce bit was encoded by Alice (local op), attacker didn't see it if they measured before her op;
            # this is illustrative only.
        # Channel noise may flip bits
        if RNG.random() < noise:
            bob_out ^= 1
        measured_alice[i] = alice_out
        measured_bob[i] = bob_out

    # Authentication: Bob compares measured_bob vs expected (if he had classical reference)
    # We'll compute Hamming distance between (measured_alice XOR nonce_bits) and measured_bob
    # If little disturbance, they match well.
    alice_encoded = measured_alice  # already encoded
    # expected parity: alice_encoded == measured_bob ideally
    mismatches = np.sum(alice_encoded != measured_bob)
    mismatch_rate = mismatches / n_pairs
    threshold = 0.15
    auth_pass = (mismatch_rate <= threshold)

    # Response token: hash of measured_bob (as classical response)
    rb_bytes = np.packbits(measured_bob)
    response_token = sha256(rb_bytes.tobytes()).hexdigest()

    return dict(
        nonce_bits=nonce_bits,
        n_pairs=n_pairs,
        measured_alice=measured_alice,
        measured_bob=measured_bob,
        mismatches=int(mismatches),
        mismatch_rate=float(mismatch_rate),
        auth_pass=auth_pass,
        disturbance=disturbance,
        response_token=response_token
    )

# ---- Visualization helpers ----
def plot_hist_counts(arr, title):
    fig, ax = plt.subplots()
    unique, counts = np.unique(arr, return_counts=True)
    ax.bar(unique.astype(str), counts)
    ax.set_title(title)
    return fig

# ---- Run simulation when requested ----
if run_button:
    st.subheader(f"Running: {protocol}")
    if protocol == "BB84 Challenge-Response":
        out = bb84_authenticate(n_qubits, noise, attacker, RNG)
        st.markdown("### Summary")
        st.write(f"Nonce (decimal): {out['nonce']}  —  (first {len(out['nonce_bits'])} nonce bits embedded)")
        st.write(f"Total qubits sent: {out['n_qubits']}")
        st.write(f"Matching bases (sifted): {out['matched_count']}")
        st.write(f"Test sample size revealed for verification: {out['test_size']}")
        st.write(f"Mismatches in test sample: {out['mismatches']}  (mismatch rate {out['mismatch_rate']*100:.2f}%)")
        if out['auth_pass']:
            st.success("Authentication PASSED (mismatch rate below threshold).")
        else:
            st.error("Authentication FAILED (mismatch rate above threshold or insufficient matched bits).")

        st.markdown("### Response token (Bob) — SHA256 of remaining sifted bits")
        st.code(out['response_token'] or "<no response bits available>")

        # Small visuals
        st.write("### Basis alignment visualization (first 100 qubits)")
        k = min(100, out['n_qubits'])
        fig, ax = plt.subplots(figsize=(10, 2))
        im = ax.imshow(
            np.vstack([out['bases_alice'][:k], out['bases_bob'][:k]]),
            aspect='auto', cmap='Greys'
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Alice bases', 'Bob bases'])
        ax.set_xlabel("qubit index (first 100)")
        st.pyplot(fig)

        st.markdown("### Note")
        st.write(
            "This BB84-style authentication uses sifting + revealing a test subset. "
            "If an attacker intercepts/resends, they introduce disturbances that increase mismatches. "
            "This is simplified and for demonstration only."
        )

    else:  # Bell-pair
        out = bell_pair_circuit_simulation(n_qubits, noise, attacker, RNG)
        st.markdown("### Summary (Bell-pair)")
        st.write(f"Pairs used: {out['n_pairs']}")
        st.write(f"Mismatches: {out['mismatches']}  (mismatch rate {out['mismatch_rate']*100:.2f}%)")
        if out['auth_pass']:
            st.success("Authentication PASSED (low mismatch rate).")
        else:
            st.error("Authentication FAILED (too many mismatches).")
        st.write(f"Disturbance events simulated (attacker action count): {out['disturbance']}")
        st.markdown("### Response token (Bob) — SHA256 of Bob's measurements")
        st.code(out['response_token'])

        st.write("### Measurement distribution (Alice vs Bob)")
        fig1 = plot_hist_counts(out['measured_alice'], "Alice measured bits counts")
        fig2 = plot_hist_counts(out['measured_bob'], "Bob measured bits counts")
        st.pyplot(fig1)
        st.pyplot(fig2)

        st.markdown("### Note")
        st.write(
            "Bell-pair based scheme here is a toy: Alice encodes nonce bits via local flips; "
            "entanglement correlations allow Bob to verify. An active eavesdropper breaks correlations, "
            "increasing mismatches. Real entanglement-based protocols are more complex."
        )

    st.markdown("---")
    st.markdown(
        """
        **Reminder:** These are simplified simulations to illustrate principles:
        - They do not perform cryptographic key management, nor do they handle replay resistance or many real-world attack vectors.
        - Real quantum authentication protocols rely on careful protocol design, classical post-processing (privacy amplification, error-correction), and proven security models.
        """
    )
else:
    st.info("Choose protocol and parameters in the sidebar, then click 'Run Simulation'.")

# ---- footer notes ----
st.markdown("---")
st.markdown("If you want, I can:")
st.markdown("- add an export button to save simulation logs / tokens;")
st.markdown("- add a classical-cryptography hybrid (e.g., sign the response with a classical MAC);")
st.markdown("- implement the full quantum circuit with PennyLane qnodes for the Bell protocol (right now the Bell protocol is a high-level stochastic sim).")
