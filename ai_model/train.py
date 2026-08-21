"""
Standalone Training & Inference Runner for ASL Signing AI Motion Generation Model.
"""

import sys
import time
import torch
import torch.optim as optim
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_model.config import AIModelConfig
from ai_model.pipeline import ASLSigningAIModel
from ai_model.dataset import ASLMotionDataset, pad_collate_fn
from ai_model.metrics import ASLMotionMetrics
from torch.utils.data import DataLoader

def run_training_demo(epochs: int = 5):
    print("=" * 70)
    print("ASL Signing AI Motion Generation Model — Training & Inference Pipeline")
    print("=" * 70)

    config = AIModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"FPS: {config.fps} | Frames: {config.max_seq_len} | Motion Vector Dim: {config.motion_dim}")
    print("-" * 70)

    # Initialize Model & Dataset
    dataset = ASLMotionDataset(config=config)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pad_collate_fn)

    model = ASLSigningAIModel(config).to(device)
    model.register_vocab(dataset.vocab)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training Loop
    print("\n[STARTING TRAINING EPOCHS]")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for tokens, target_motion, mask in dataloader:
            tokens, target_motion = tokens.to(device), target_motion.to(device)
            mask = mask.to(device) if mask is not None else None

            optimizer.zero_grad()

            pred_motion, smplx_dict, vel, acc = model(tokens, mask=mask, target_seq_len=config.max_seq_len)
            loss_dict = model.loss_fn(pred_motion, target_motion, smplx_dict, vel, acc)

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        elapsed = time.time() - start_t
        avg_loss = total_loss / len(dataloader)

        print(f"  Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Hand Loss: {loss_dict['l_hand']:.4f} | NMM Loss: {loss_dict['l_nmm']:.4f} | Time: {elapsed:.2f}s", flush=True)

    # Evaluation
    print("\n[EVALUATING MODEL METRICS]")
    model.eval()
    with torch.no_grad():
        for tokens, target_motion, mask in dataloader:
            tokens, target_motion = tokens.to(device), target_motion.to(device)
            pred_motion, smplx_dict, _, _ = model(tokens, target_seq_len=config.max_seq_len)
            eval_results = ASLMotionMetrics.evaluate(pred_motion, target_motion, smplx_dict)
            break

    print(f"  MPJPE Error:               {eval_results['mpjpe']} rad/mm")
    print(f"  Hand Joint Error:          {eval_results['hand_joint_error']} rad/mm")
    print(f"  Velocity Smoothness:       {eval_results['velocity_smoothness']}")
    print(f"  Acceleration Smoothness:   {eval_results['acceleration_smoothness']}")
    print(f"  Jerk Metric:               {eval_results['jerk_metric']}")

    # Inference Demo
    print("\n" + "=" * 70)
    print("[RUNNING END-TO-END INFERENCE DEMO]")
    print("=" * 70)

    test_sentences = [
        "WHAT IS YOUR NAME?",
        "GOOD MORNING",
        "YESTERDAY I WENT TO SCHOOL",
        "THANK-YOU VERY-MUCH"
    ]

    for sentence in test_sentences:
        t0 = time.time()
        export = model.generate_motion_from_gloss(sentence, target_seq_len=120, device=device)
        latency = (time.time() - t0) * 1000

        print(f"\n> Input Sentence: '{sentence}'")
        print(f"  Tokens:           {export['tokens']}")
        print(f"  Frames Generated: {export['frame_count']} frames ({export['duration']}s at {export['fps']} FPS)")
        print(f"  Latency:          {latency:.2f} ms")
        print(f"  Body Pose Dims:   {len(export['frames'][0]['body_pose'])} (21 joints)")
        print(f"  Right Hand Dims:  {len(export['frames'][0]['right_hand_pose'])} (15 finger joints)")
        print(f"  Left Hand Dims:   {len(export['frames'][0]['left_hand_pose'])} (15 finger joints)")
        print(f"  Facial Morph Dims:{len(export['frames'][0]['expression'])} blendshapes")

    print("\n[SUCCESS] AI Motion Generation Pipeline successfully trained and verified!")

if __name__ == "__main__":
    run_training_demo(epochs=5)
