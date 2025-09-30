import argparse, time, random
from pylsl import StreamInfo, StreamOutlet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rsvp","ssvep"], default="rsvp")
    ap.add_argument("--rate_hz", type=float, default=8.0)
    args = ap.parse_args()

    info = StreamInfo("Markers", "Markers", 1, 0, "string", "sim-markers")
    outlet = StreamOutlet(info)

    isi = 1.0/args.rate_hz
    vocab = [f"id_{i:03d}" for i in range(1,101)]
    ssvep = ["31.0","32.0","33.0","34.5"]
    print(f"Streaming markers for {args.task} at {args.rate_hz}Hz")
    while True:
        token = random.choice(vocab) if args.task=="rsvp" else random.choice(ssvep)
        outlet.push_sample([token]); time.sleep(isi)

if __name__ == "__main__":
    main()
