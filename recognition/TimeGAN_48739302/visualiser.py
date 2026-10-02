import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

# LOBSTER configuration
PRICE_SCALE = 10000.0

EVENT_TYPES = {
    1: "Submission",
    2: "Cancellation",
    3: "Deletion",
    4: "Visible Execution",
    5: "Hidden Execution",
    7: "Trading Halt",
}


# Loading
def load_lobster(message_file, orderbook_file):
    """
    Load LOBSTER message and orderbook files.

    Message:
        time, type, order_id, size, price, direction

    Orderbook:
        ask_price_1, ask_size_1,
        bid_price_1, bid_size_1,
        ...
    """

    message = pd.read_csv(
        message_file,
        header=None,
        names=[
            "time",
            "type",
            "order_id",
            "size",
            "price",
            "direction",
        ],
    )

    orderbook = pd.read_csv(
        orderbook_file,
        header=None,
    )

    if len(message) != len(orderbook):
        raise ValueError(
            f"Message rows ({len(message)}) != "
            f"orderbook rows ({len(orderbook)})"
        )

    n_columns = orderbook.shape[1]

    if n_columns % 4 != 0:
        raise ValueError(
            f"Orderbook has {n_columns} columns, "
            "which is not divisible by 4."
        )

    levels = n_columns // 4

    print(f"Loaded {len(message):,} events")
    print(f"Detected {levels} order-book levels")

    return message, orderbook, levels


# ============================================================
# Utilities
# ============================================================


def format_time(seconds):
    """
    Convert seconds after midnight into HH:MM:SS.mmm
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def get_book_row(orderbook, row, levels):
    """
    Extract one order-book state.

    Returns:
        ask_prices
        ask_sizes
        bid_prices
        bid_sizes
    """

    data = orderbook.iloc[row].to_numpy(dtype=float)

    ask_prices = data[0::4][:levels] / PRICE_SCALE
    ask_sizes = data[1::4][:levels]

    bid_prices = data[2::4][:levels] / PRICE_SCALE
    bid_sizes = data[3::4][:levels]

    return (
        ask_prices,
        ask_sizes,
        bid_prices,
        bid_sizes,
    )


def clean_prices(prices):
    """
    Remove LOBSTER dummy prices.
    """

    prices = np.asarray(prices, dtype=float)

    valid = np.isfinite(prices)

    # LOBSTER dummy values become enormous after /10000
    valid &= np.abs(prices) < 1e6

    return np.where(valid, prices, np.nan)


def book_statistics(orderbook, levels):
    """
    Calculate mid-price, spread and imbalance
    for every order-book row.
    """

    n = len(orderbook)

    mid = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    imbalance = np.full(n, np.nan)

    for i in range(n):

        ask_p, ask_s, bid_p, bid_s = get_book_row(
            orderbook,
            i,
            levels,
        )

        ask_p = clean_prices(ask_p)
        bid_p = clean_prices(bid_p)

        valid_ask = np.isfinite(ask_p) & (ask_s > 0)
        valid_bid = np.isfinite(bid_p) & (bid_s > 0)

        if np.any(valid_ask) and np.any(valid_bid):

            best_ask = ask_p[valid_ask][0]
            best_bid = bid_p[valid_bid][0]

            mid[i] = (best_ask + best_bid) / 2.0
            spread[i] = best_ask - best_bid

        total_ask = np.sum(ask_s[valid_ask])
        total_bid = np.sum(bid_s[valid_bid])

        total_volume = total_bid + total_ask

        if total_volume > 0:
            imbalance[i] = (total_bid - total_ask) / total_volume

    return mid, spread, imbalance


# Static plots
def plot_overview(message, orderbook, levels):
    """
    Plot the main time-series statistics.
    """

    time = message["time"].to_numpy()

    mid, spread, imbalance = book_statistics(
        orderbook,
        levels,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
    )

    # Mid price
    axes[0].plot(
        time,
        mid,
        linewidth=1,
    )

    axes[0].set_ylabel("Mid Price ($)")
    axes[0].set_title("LOBSTER Order Book")

    axes[0].grid(True)

    # Spread
    axes[1].plot(
        time,
        spread,
        linewidth=1,
    )

    axes[1].set_ylabel("Spread ($)")
    axes[1].grid(True)

    # Imbalance
    axes[2].plot(
        time,
        imbalance,
        linewidth=1,
    )

    axes[2].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axes[2].set_ylabel("Imbalance")
    axes[2].set_xlabel("Seconds after midnight")
    axes[2].set_ylim(-1, 1)

    axes[2].grid(True)

    plt.tight_layout()

    return fig


# Event statistics
def plot_event_types(message):
    """
    Plot the distribution of LOBSTER event types.
    """

    counts = message["type"].value_counts().sort_index()

    labels = [
        EVENT_TYPES.get(
            int(event_type),
            f"Unknown ({event_type})",
        )
        for event_type in counts.index
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        labels,
        counts.values,
    )

    ax.set_ylabel("Number of Events")
    ax.set_title("LOBSTER Message Event Distribution")

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    return fig


# Interactive order book
def interactive_book(message, orderbook, levels):
    """
    Interactive order-book visualiser.

    A slider allows the user to move through individual
    LOBSTER events.
    """

    n = len(message)

    initial_row = 0

    fig = plt.figure(figsize=(15, 10))

    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[3, 1.5, 0.7],
    )

    ax_book = fig.add_subplot(gs[0, :])
    ax_price = fig.add_subplot(gs[1, 0])
    ax_imbalance = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[2, :])

    # Statistics for lower plots
    time = message["time"].to_numpy()

    mid, spread, imbalance = book_statistics(
        orderbook,
        levels,
    )

    # Time-series plots
    ax_price.plot(
        time,
        mid,
        linewidth=1,
    )

    current_price_line = ax_price.axvline(
        time[initial_row],
        linestyle="--",
    )

    ax_price.set_title("Mid Price")
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Price ($)")
    ax_price.grid(True)

    ax_imbalance.plot(
        time,
        imbalance,
        linewidth=1,
    )

    current_imbalance_line = ax_imbalance.axvline(
        time[initial_row],
        linestyle="--",
    )

    ax_imbalance.set_title("Order Book Imbalance")
    ax_imbalance.set_xlabel("Time")
    ax_imbalance.set_ylabel("Imbalance")
    ax_imbalance.set_ylim(-1, 1)
    ax_imbalance.grid(True)

    # Order book drawing
    def draw_book(row):

        ax_book.clear()

        ask_p, ask_s, bid_p, bid_s = get_book_row(
            orderbook,
            row,
            levels,
        )

        ask_p = clean_prices(ask_p)
        bid_p = clean_prices(bid_p)

        valid_ask = np.isfinite(ask_p) & (ask_s > 0)

        valid_bid = np.isfinite(bid_p) & (bid_s > 0)

        ask_p = ask_p[valid_ask]
        ask_s = ask_s[valid_ask]

        bid_p = bid_p[valid_bid]
        bid_s = bid_s[valid_bid]

        # Only show available levels
        if len(ask_p) == 0 and len(bid_p) == 0:
            ax_book.set_title("Empty Order Book")
            return

        # Plot asks
        if len(ask_p) > 0:

            ax_book.barh(
                ask_p,
                ask_s,
                height=0.5
                * np.maximum(
                    np.median(np.diff(ask_p)) if len(ask_p) > 1 else 0.01,
                    0.001,
                ),
                alpha=0.7,
                label="Ask",
            )

        # Plot bids
        if len(bid_p) > 0:

            ax_book.barh(
                bid_p,
                bid_s,
                height=0.5
                * np.maximum(
                    np.median(np.diff(bid_p)) if len(bid_p) > 1 else 0.01,
                    0.001,
                ),
                alpha=0.7,
                label="Bid",
            )

        # Best bid / ask
        best_ask = ask_p[0] if len(ask_p) else np.nan
        best_bid = bid_p[0] if len(bid_p) else np.nan

        if np.isfinite(best_ask):
            ax_book.axhline(
                best_ask,
                linestyle="--",
                linewidth=1,
            )

        if np.isfinite(best_bid):
            ax_book.axhline(
                best_bid,
                linestyle="--",
                linewidth=1,
            )

        # Labels
        event_type = int(message.iloc[row]["type"])

        event_name = EVENT_TYPES.get(
            event_type,
            f"Unknown ({event_type})",
        )

        timestamp = message.iloc[row]["time"]

        ax_book.set_title(
            f"Order Book @ "
            f"{format_time(timestamp)} "
            f"| Event: {event_name}"
        )

        ax_book.set_xlabel("Volume")
        ax_book.set_ylabel("Price ($)")

        ax_book.legend()
        ax_book.grid(True)

        # Display message
        msg = message.iloc[row]

        ax_info.clear()
        ax_info.axis("off")

        info = (
            f"ROW: {row:,} / {n - 1:,}    "
            f"TIME: {format_time(msg['time'])}    "
            f"TYPE: {event_type} ({event_name})    "
            f"ORDER ID: {int(msg['order_id'])}    "
            f"SIZE: {int(msg['size']):,}    "
            f"PRICE: ${msg['price'] / PRICE_SCALE:.4f}    "
            f"DIRECTION: {int(msg['direction'])}"
        )

        ax_info.text(
            0.01,
            0.5,
            info,
            fontsize=11,
            verticalalignment="center",
        )

        # Time markers
        current_price_line.set_xdata([timestamp, timestamp])

        current_imbalance_line.set_xdata([timestamp, timestamp])

        # ----------------------------------------------------
        # Highlight current point
        # ----------------------------------------------------

        fig.canvas.draw_idle()

    # Slider
    slider_ax = fig.add_axes([0.15, 0.02, 0.70, 0.03])

    slider = Slider(
        slider_ax,
        "Event",
        0,
        n - 1,
        valinit=initial_row,
        valstep=1,
    )

    def update(value):
        row = int(slider.val)
        draw_book(row)

    slider.on_changed(update)

    draw_book(initial_row)

    plt.show()


def main():

    parser = argparse.ArgumentParser(
        description="Visualise LOBSTER order book data."
    )

    parser.add_argument(
        "message",
        type=Path,
        help="LOBSTER message CSV",
    )

    parser.add_argument(
        "orderbook",
        type=Path,
        help="LOBSTER orderbook CSV",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open interactive order-book viewer",
    )

    parser.add_argument(
        "--events",
        action="store_true",
        help="Plot event-type distribution",
    )

    parser.add_argument(
        "--overview",
        action="store_true",
        help="Plot mid-price, spread and imbalance",
    )

    args = parser.parse_args()

    message, orderbook, levels = load_lobster(
        args.message,
        args.orderbook,
    )

    # Default behaviour
    if not (args.interactive or args.events or args.overview):
        args.interactive = True
        args.overview = True
        args.events = True

    if args.overview:
        plot_overview(
            message,
            orderbook,
            levels,
        )

    if args.events:
        plot_event_types(message)

    if args.interactive:
        interactive_book(
            message,
            orderbook,
            levels,
        )

    plt.show()


if __name__ == "__main__":
    main()
