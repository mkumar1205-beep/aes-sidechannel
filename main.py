import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]

def hamming_weight(x):
    return bin(x).count('1')

TRACE_LENGTH = 50   # time samples per trace
LEAK_POINT   = 10   # sample index where leakage occurs
NOISE_STD    = 1.5  # Gaussian noise standard deviation

def simulate_trace(plaintext_byte: int, key_byte: int, noise_std: float = NOISE_STD) -> np.ndarray:
    trace = np.random.normal(0, noise_std, TRACE_LENGTH)
    sbox_out = SBOX[plaintext_byte ^ key_byte]
    hw = hamming_weight(sbox_out)
    # Leakage: power spike at LEAK_POINT proportional to HW
    trace[LEAK_POINT]   += hw * 1.0
    trace[LEAK_POINT-1] += hw * 0.3   # pre-charge
    trace[LEAK_POINT+1] += hw * 0.3   # discharge
    return trace

def collect_traces(true_key_byte: int, num_traces: int, noise_std: float = NOISE_STD):
    plaintexts = np.random.randint(0, 256, num_traces, dtype=np.uint8)
    traces = np.array([simulate_trace(int(p), true_key_byte, noise_std) for p in plaintexts])
    return plaintexts, traces

def cpa_attack(plaintexts, traces):
    print("\n[CPA] Running Correlation Power Analysis...")
    num_traces, trace_len = traces.shape
    correlations = np.zeros((256, trace_len))

    for k in range(256):
        hw_model = np.array([hamming_weight(SBOX[int(p) ^ k]) for p in plaintexts], dtype=float)
        for t in range(trace_len):
            # Pearson correlation between power model and measured power
            correlations[k, t] = np.corrcoef(hw_model, traces[:, t])[0, 1]

    max_corr_per_key = np.max(np.abs(correlations), axis=1)
    recovered_key = np.argmax(max_corr_per_key)
    return recovered_key, correlations, max_corr_per_key

def ml_attack(plaintexts_train, traces_train, plaintexts_test, traces_test, true_key_byte):
    """
    Train an MLP to directly classify key byte from (plaintext, trace) pairs.
    Labels: HW class of SBOX[plaintext XOR true_key] (9 classes: HW 0..8)
    At test time, try all 256 key hypotheses — correct key gives best accuracy.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("\n[ML] PyTorch not installed. Run: pip install torch")
        print("[ML] Skipping ML attack, showing CPA results only.\n")
        return None

    print("\n[ML] Training Neural Network for Key Recovery...")

    
    def make_features(pts, trs):
        pt_feat = pts.reshape(-1,1).astype(np.float32) / 255.0
        return np.hstack([pt_feat, trs.astype(np.float32)])

    X_train = make_features(plaintexts_train, traces_train)
    X_test  = make_features(plaintexts_test,  traces_test)

    y_train = np.array([hamming_weight(SBOX[int(p)^true_key_byte]) for p in plaintexts_train])
    y_test  = np.array([hamming_weight(SBOX[int(p)^true_key_byte]) for p in plaintexts_test])

    # MLP model
    class PowerMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 9),          # 9 HW classes
            )
        def forward(self, x): return self.net(x)

    input_dim = X_train.shape[1]
    model = PowerMLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training
    X_t = torch.tensor(X_train); y_t = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(30):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}/30 | Loss: {total_loss/len(loader):.4f}")

    # Key recovery: test all 256 key hypotheses
    model.eval()
    key_scores = []
    with torch.no_grad():
        for k in range(256):
            y_hyp = np.array([hamming_weight(SBOX[int(p)^k]) for p in plaintexts_test])
            X_te  = torch.tensor(make_features(plaintexts_test, traces_test))
            logits = model(X_te)
            probs  = torch.softmax(logits, dim=1).numpy()
            # Score: average probability assigned to the hypothesized HW class
            score = np.mean([probs[i, y_hyp[i]] for i in range(len(y_hyp))])
            key_scores.append(score)

    recovered_key_ml = np.argmax(key_scores)
    return recovered_key_ml, np.array(key_scores)

def plot_results(true_key, correlations, max_corr, ml_scores=None, traces=None, plaintexts=None):
    cols = 2 if ml_scores is None else 3
    has_traces = traces is not None
    rows = 2 if has_traces else 1

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    fig.patch.set_facecolor('#0d1117')

    if rows == 1:
        axes = [axes]

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color='white', fontsize=11, pad=8)
        ax.set_xlabel(xlabel, color='#8b949e')
        ax.set_ylabel(ylabel, color='#8b949e')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # Row 0: CPA correlation over time
    ax = axes[0][0]
    for k in range(256):
        color = '#f85149' if k == true_key else '#21262d'
        lw    = 2.0 if k == true_key else 0.3
        ax.plot(np.abs(correlations[k]), color=color, linewidth=lw, alpha=0.8 if k==true_key else 0.5)
    ax.axvline(LEAK_POINT, color='#f0e68c', linestyle='--', alpha=0.7, label='Leak point')
    style_ax(ax, 'CPA: Correlation Traces (all 256 key guesses)', 'Time Sample', '|Correlation|')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)

    # Row 0: Max correlation per key
    ax = axes[0][1]
    colors_bar = ['#f85149' if k==true_key else '#388bfd' for k in range(256)]
    ax.bar(range(256), max_corr, color=colors_bar, width=1.0)
    ax.axvline(true_key, color='#f85149', linestyle='--', lw=1.5)
    style_ax(ax, f'CPA: Max |Correlation| per Key Hypothesis\nTrue key = 0x{true_key:02X} (red)', 'Key Byte (0-255)', 'Max |Correlation|')

    # Row 0: ML scores
    if ml_scores is not None:
        ax = axes[0][2]
        colors_ml = ['#f85149' if k==true_key else '#3fb950' for k in range(256)]
        ax.bar(range(256), ml_scores, color=colors_ml, width=1.0)
        ax.axvline(true_key, color='#f85149', linestyle='--', lw=1.5)
        style_ax(ax, f'ML: Key Score per Hypothesis\nTrue key = 0x{true_key:02X} (red)', 'Key Byte (0-255)', 'Avg HW Probability')

    # Row 1: Sample traces
    if has_traces:
        ax = axes[1][0]
        for i in range(min(10, len(traces))):
            ax.plot(traces[i], alpha=0.5, linewidth=0.8)
        ax.axvline(LEAK_POINT, color='#f0e68c', linestyle='--', alpha=0.8, label='Leak point')
        style_ax(ax, 'Sample Power Traces (10 shown)', 'Time Sample', 'Power (a.u.)')
        ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)

        # HW distribution
        ax = axes[1][1]
        hw_vals = [hamming_weight(SBOX[int(p)^true_key]) for p in plaintexts[:500]]
        ax.hist(hw_vals, bins=9, range=(0,9), color='#388bfd', edgecolor='#0d1117', rwidth=0.8)
        style_ax(ax, 'HW Distribution: SBOX[plaintext XOR key]', 'Hamming Weight', 'Count')

        if ml_scores is not None and cols > 2:
            ax = axes[1][2]
            cpa_rank  = sorted(range(256), key=lambda k: -max_corr[k]).index(true_key)
            ml_rank   = sorted(range(256), key=lambda k: -ml_scores[k]).index(true_key)
            methods   = ['CPA\n(Classical)', 'MLP\n(Deep Learning)']
            ranks     = [cpa_rank+1, ml_rank+1]
            bar_colors= ['#388bfd','#3fb950']
            bars = ax.bar(methods, ranks, color=bar_colors, width=0.4)
            for bar, rank in zip(bars, ranks):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'Rank {rank}',
                        ha='center', color='white', fontsize=10)
            style_ax(ax, 'Key Recovery Rank\n(Rank 1 = Perfect Recovery)', 'Method', 'Rank of True Key')
            ax.set_ylim(0, max(ranks)*1.3 + 1)

    fig.suptitle('AES Side-Channel Attack: CPA vs ML\nIIT Bombay + IIT Hyderabad Summer School Project',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/side_channel_results.png', dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    print("\n  📊 Plot saved → side_channel_results.png")
    plt.close()

def run_attack():
    print("\n" + "="*55)
    print("  PROJECT 2: POWER SIDE-CHANNEL ATTACK + ML")
    print("  IIT Bombay (side-channel) + IIT Hyderabad (ML crypto)")
    print("="*55)

    TRUE_KEY_BYTE = 0xAB   
    NUM_TRAIN     = 2000
    NUM_TEST      = 500
    NOISE         = 1.5

    print(f"\n  True key byte:  0x{TRUE_KEY_BYTE:02X}")
    print(f"  Training traces: {NUM_TRAIN}")
    print(f"  Test traces:     {NUM_TEST}")
    print(f"  Noise std:       {NOISE}")

    # Collect traces
    pts_train, traces_train = collect_traces(TRUE_KEY_BYTE, NUM_TRAIN, NOISE)
    pts_test,  traces_test  = collect_traces(TRUE_KEY_BYTE, NUM_TEST,  NOISE)

    # CPA Attack
    cpa_key, correlations, max_corr = cpa_attack(pts_train, traces_train)
    cpa_rank = sorted(range(256), key=lambda k: -max_corr[k]).index(TRUE_KEY_BYTE) + 1
    print(f"\n  [CPA] Recovered key: 0x{cpa_key:02X}  (true: 0x{TRUE_KEY_BYTE:02X})")
    print(f"  [CPA] True key rank: {cpa_rank}/256")
    print(f"  [CPA] {'CORRECT!' if cpa_key==TRUE_KEY_BYTE else 'Wrong (try more traces)'}")

    # ML Attack
    ml_result = ml_attack(pts_train, traces_train, pts_test, traces_test, TRUE_KEY_BYTE)

    if ml_result:
        ml_key, ml_scores = ml_result
        ml_rank = sorted(range(256), key=lambda k: -ml_scores[k]).index(TRUE_KEY_BYTE) + 1
        print(f"\n  [ML]  Recovered key: 0x{ml_key:02X}  (true: 0x{TRUE_KEY_BYTE:02X})")
        print(f"  [ML]  True key rank: {ml_rank}/256")
        print(f"  [ML]  {'CORRECT!' if ml_key==TRUE_KEY_BYTE else 'Wrong (try more traces)'}")
    else:
        ml_scores = None

    # Plot
    plot_results(TRUE_KEY_BYTE, correlations, max_corr, ml_scores, traces_train[:100], pts_train[:100])

    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    print("  • Power traces leak via Hamming weight of S-Box output")
    print("  • CPA exploits statistical correlation (classical attack)")
    print("  • MLP learns leakage patterns end-to-end")
    print("  • Both recover the secret key without knowing the algorithm")
    print("  • Countermeasure: masking, noise injection, constant-time code")
    print("="*55 + "\n")

if __name__ == "__main__":
    np.random.seed(42)
    run_attack()