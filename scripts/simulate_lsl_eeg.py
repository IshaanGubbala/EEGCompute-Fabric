import argparse, time, math, random
from pylsl import StreamInfo, StreamOutlet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=int, default=512)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--name", type=str, default="EEG")
    ap.add_argument("--type", type=str, default="EEG")
    ap.add_argument("--alpha_hz", type=float, default=10.0)
    args = ap.parse_args()

    info = StreamInfo(args.name, args.type, args.channels, args.fs, "float32", "sim-eeg")
    outlet = StreamOutlet(info)

    t = 0.0; dt = 1.0/args.fs
    print(f"Streaming {args.channels}ch @ {args.fs}Hz on LSL '{args.name}'")
    while True:
        row = [20.0*math.sin(2*math.pi*args.alpha_hz*t + ch*0.5) + random.gauss(0, 5.0) for ch in range(args.channels)]
        outlet.push_sample(row); t += dt; time.sleep(dt)

if __name__ == "__main__":
    main()
