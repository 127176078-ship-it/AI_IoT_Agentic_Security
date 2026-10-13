import argparse
import warnings

warnings.filterwarnings(
    "ignore"
)

from scapy.all import sniff


def capture_packets(
    count=10,
    timeout=10,
    interface=None,
):

    kwargs = {
        "count": count,
        "timeout": timeout,
    }

    if interface:
        kwargs["iface"] = interface

    packets = sniff(
        **kwargs
    )

    return packets


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--interface",
        default=None,
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SCAPY LIVE CAPTURE")
    print("=" * 60)

    try:

        packets = capture_packets(
            args.count,
            args.timeout,
            args.interface,
        )

        print(
            f"Captured packets: {len(packets)}"
        )

        for index, packet in enumerate(
            packets[:5],
            start=1,
        ):

            print(
                f"{index}. {packet.summary()}"
            )

        return 0

    except Exception as exc:

        print(
            "[ERROR] Packet capture unavailable."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "UNSW-NB15 ML analysis remains available."
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )