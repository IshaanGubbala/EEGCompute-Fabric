#!/usr/bin/env python3
"""Real-time re-ranking demonstration using P300 scores from Fabric bus.

This script demonstrates the core value proposition of EEGCompute Fabric:
brain-guided re-ranking of items based on P300 responses.

The demo:
1. Creates a toy dataset of 100 items with initial scores
2. Subscribes to live P300 scores from the Fabric bus
3. Updates item rankings when new P300 scores arrive
4. Shows real-time movement in the top-10 rankings

Usage:
    # Start the full pipeline first:
    python scripts/run_all_v11.py

    # Then in another terminal:
    source .venv/bin/activate
    python scripts/reranking_demo.py

The demo will show how P300 scores from brain signals can improve
ranking quality by identifying truly relevant items.
"""

import sys
import time
import asyncio
import requests
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app_logic.reranker.ranker import rerank

class ReRankingDemo:
    """Demonstrates real-time re-ranking with P300 scores."""

    def __init__(self, fabric_url="http://localhost:8008"):
        self.fabric_url = fabric_url
        self.p300_scores = {}  # item_id -> cumulative P300 score
        self.score_counts = {}  # item_id -> number of P300 observations

        # Create toy dataset
        self.items = self._create_toy_dataset()
        print(f"📊 Created toy dataset with {len(self.items)} items")

        # Track statistics
        self.total_updates = 0
        self.last_update_time = time.time()

    def _create_toy_dataset(self) -> List[Tuple[str, float]]:
        """Create a toy dataset with 100 items and initial relevance scores."""
        items = []

        # Category 1: High-relevance items (should rank higher after P300)
        high_relevance_items = [
            ("target_image_001", 0.65),
            ("target_image_007", 0.68),
            ("target_image_015", 0.62),
            ("target_image_023", 0.66),
            ("target_image_042", 0.64),
        ]

        # Category 2: Medium-relevance items
        medium_relevance_items = [
            (f"medium_item_{i:03d}", random.uniform(0.45, 0.65))
            for i in range(20)
        ]

        # Category 3: Low-relevance items (noise)
        low_relevance_items = [
            (f"distractor_{i:03d}", random.uniform(0.15, 0.45))
            for i in range(75)
        ]

        items.extend(high_relevance_items)
        items.extend(medium_relevance_items)
        items.extend(low_relevance_items)

        # Shuffle to simulate real-world disorder
        random.shuffle(items)
        return items

    def get_current_rankings(self) -> List[Tuple[str, float]]:
        """Get current item rankings using P300 scores."""
        return rerank(self.items, self.p300_scores, alpha=0.3)

    def fetch_latest_scores(self) -> Dict:
        """Fetch latest scores from Fabric bus."""
        try:
            response = requests.get(f"{self.fabric_url}/latest", timeout=2.0)
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            print(f"⚠️  Error fetching scores: {e}")
            return {}

    def update_p300_scores(self, fabric_data: Dict):
        """Update P300 scores from Fabric bus data."""
        if 'p300' not in fabric_data:
            return False

        p300_data = fabric_data['p300']
        item_id = p300_data.get('meta', {}).get('item_id')
        score = p300_data.get('value', 0.0)
        confidence = p300_data.get('confidence', 0.0)

        if not item_id or confidence < 0.3:  # Filter low-confidence scores
            return False

        # Update cumulative scores (simple averaging)
        if item_id in self.p300_scores:
            current_count = self.score_counts[item_id]
            current_score = self.p300_scores[item_id]

            # Running average
            new_count = current_count + 1
            new_score = (current_score * current_count + score) / new_count

            self.p300_scores[item_id] = new_score
            self.score_counts[item_id] = new_count
        else:
            self.p300_scores[item_id] = score
            self.score_counts[item_id] = 1

        self.total_updates += 1
        self.last_update_time = time.time()

        print(f"🧠 Updated P300 score for {item_id}: {score:.3f} "
              f"(avg: {self.p300_scores[item_id]:.3f}, n={self.score_counts[item_id]})")

        return True

    def print_top_rankings(self, n=10):
        """Print current top-N rankings."""
        rankings = self.get_current_rankings()
        print(f"\n🏆 Current Top-{n} Rankings:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Item ID':<20} {'Score':<8} {'P300 Boost':<12} {'Total':<8}")
        print("-" * 70)

        for i, (item_id, total_score) in enumerate(rankings[:n], 1):
            prior_score = dict(self.items)[item_id]
            p300_boost = self.p300_scores.get(item_id, 0.0) * 0.3  # alpha=0.3
            print(f"{i:<5} {item_id:<20} {prior_score:<8.3f} {p300_boost:<12.3f} {total_score:<8.3f}")

    def print_statistics(self):
        """Print demo statistics."""
        print(f"\n📊 Demo Statistics:")
        print(f"   Total P300 updates: {self.total_updates}")
        print(f"   Items with P300 scores: {len(self.p300_scores)}")
        print(f"   Last update: {time.time() - self.last_update_time:.1f}s ago")

        if self.p300_scores:
            avg_score = sum(self.p300_scores.values()) / len(self.p300_scores)
            max_score = max(self.p300_scores.values())
            print(f"   Average P300 score: {avg_score:.3f}")
            print(f"   Maximum P300 score: {max_score:.3f}")

    def print_value_demonstration(self):
        """Show the value of P300-guided re-ranking."""
        original_rankings = sorted(self.items, key=lambda x: x[1], reverse=True)
        current_rankings = self.get_current_rankings()

        print(f"\n🎯 Re-ranking Value Demonstration:")
        print("-" * 50)

        # Find target items in rankings
        target_items = [item for item in self.items if item[0].startswith("target_image_")]

        print(f"Target Items Performance:")
        for item_id, prior_score in target_items:
            # Find positions
            orig_pos = next((i for i, (iid, _) in enumerate(original_rankings, 1) if iid == item_id), None)
            curr_pos = next((i for i, (iid, _) in enumerate(current_rankings, 1) if iid == item_id), None)

            if orig_pos and curr_pos:
                movement = orig_pos - curr_pos  # positive = moved up
                p300_score = self.p300_scores.get(item_id, 0.0)
                status = "📈" if movement > 0 else "📉" if movement < 0 else "➡️"

                print(f"  {item_id}: {orig_pos} → {curr_pos} ({movement:+d}) "
                      f"P300: {p300_score:.3f} {status}")

    async def run_demo(self):
        """Run the real-time re-ranking demonstration."""
        print("🚀 Starting Real-Time Re-ranking Demo")
        print("=" * 60)

        print(f"🔗 Connecting to Fabric bus: {self.fabric_url}")
        print("🎯 Waiting for P300 scores to demonstrate brain-guided re-ranking...")
        print("💡 Start the RSVP demo to see items being re-ranked!")
        print("\n⏹️  Press Ctrl+C to stop\n")

        # Initial rankings
        self.print_top_rankings()

        last_display_time = time.time()
        display_interval = 5.0  # Update display every 5 seconds

        try:
            while True:
                # Fetch latest scores
                fabric_data = self.fetch_latest_scores()

                # Update P300 scores if new data available
                if fabric_data:
                    updated = self.update_p300_scores(fabric_data)

                    if updated:
                        # Show immediate ranking update
                        self.print_top_rankings()
                        self.print_value_demonstration()

                # Periodic display update
                current_time = time.time()
                if current_time - last_display_time > display_interval:
                    if not fabric_data or 'p300' not in fabric_data:
                        print("⏳ Waiting for P300 scores...")
                        print("💭 TIP: Make sure the RSVP demo is running!")

                    self.print_statistics()
                    last_display_time = current_time

                await asyncio.sleep(0.5)  # Check for updates every 500ms

        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        # Final statistics
        print("\n" + "=" * 60)
        print("📊 Final Demo Results:")
        self.print_top_rankings()
        self.print_value_demonstration()
        self.print_statistics()

        if self.total_updates > 0:
            print("\n✅ Demo completed successfully!")
            print("🎉 Brain-guided re-ranking demonstrated with live P300 scores!")
        else:
            print("\n⚠️  No P300 scores received during demo")
            print("💡 Make sure the Fabric bus and RSVP demo are running")

async def main():
    """Main entry point for the re-ranking demo."""
    demo = ReRankingDemo()
    await demo.run_demo()

if __name__ == "__main__":
    print("🧠 EEGCompute Fabric - Real-Time Re-ranking Demo")
    print("Demonstrating brain-guided item ranking with P300 signals")
    print("-" * 60)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)