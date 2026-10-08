"""
Nabhadrishti (नभदृष्टि) - BAS Payload Operations Copilot
=========================================================
Usage:
    python run.py                       # auto-detect camera (webcam/DroidCam)
    python run.py --source 0            # explicit webcam index
    python run.py --source http://192.168.1.7:4747/video   # DroidCam over Wi-Fi
    python run.py --source demo.mp4     # looped video file fallback

Then open http://localhost:8000 in a browser.
"""

import argparse
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Nabhadrishti - BAS Payload Operations Copilot")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--source", default=None,
        help="Camera source: index (0), http URL (DroidCam), rtsp URL, "
             "or video file path. Also settable via BAS_CAMERA_SOURCE env var.")
    parser.add_argument(
        "--protocol", default="flask",
        choices=["flask", "sample", "gesture"],
        help="Experiment protocol: 'flask' (reagent transfer demo), "
             "'sample' (SIH 26174 red/yellow box experiment), or "
             "'gesture' (hand-gesture mission: palm/peace/wave/gun/fist). "
             "Also settable via BAS_PROTOCOL env var.")
    parser.add_argument(
        "--droidcam", action="store_true",
        help="Auto-discover a DroidCam phone on the Wi-Fi network "
             "(scans port 4747, ~3 s) and use it as the camera. "
             "Open the DroidCam app on the phone first. Explicit "
             "--source overrides this.")
    parser.add_argument(
        "--wide", action="store_true",
        help="Open-vocabulary detection (YOLO-World): recognizes pens, "
             "pencils, spectacles, headphones, tools and ~60 more concepts "
             "zero-shot. Customize via BAS_OPEN_VOCAB_CLASSES='a,b,c'. "
             "First run downloads the model (~50 MB); lower FPS on CPU.")
    args = parser.parse_args()

    if args.source is not None:
        os.environ["BAS_CAMERA_SOURCE"] = args.source
    os.environ.setdefault("BAS_PROTOCOL", args.protocol)
    if args.wide:
        os.environ["BAS_OPEN_VOCAB"] = "1"
    if args.droidcam and args.source is None and "BAS_CAMERA_SOURCE" not in os.environ:
        from backend.camera import find_droidcam, find_virtual_camera_index
        url = find_droidcam()
        if url:
            os.environ["BAS_CAMERA_SOURCE"] = url
            print(f"DroidCam discovered: {url}")
        else:
            # Newer DroidCam versions need their PC client; once it is
            # connected the phone appears as a local virtual webcam.
            idx = find_virtual_camera_index("droidcam")
            if idx is not None:
                os.environ["BAS_CAMERA_SOURCE"] = str(idx)
                print(f"DroidCam virtual webcam found at camera index {idx}")
            else:
                print("No DroidCam found - falling back to local webcam "
                      "auto-probe. Checks: phone app open, same Wi-Fi, or "
                      "install the DroidCam Windows client and connect.")

    uvicorn.run(
        "backend.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
